import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper as nh

np.random.seed(1234)
H=8; NH=2; HS=4; V=16; L=1
assert NH*HS==H
EMB_OUT="/model/embed_tokens/Gather/output_0"
OUT="test/models/pipeline-model-tiny"

def init(name, arr):
    return nh.from_array(arr.astype(np.float32), name=name)

def save(model, path):
    model.opset_import[0].version = 17
    onnx.checker.check_model(model)
    onnx.save(model, path)

# ---- embeds.onnx ----
emb_w = (np.random.randn(V,H)*0.2).astype(np.float32)
g = helper.make_graph(
    [helper.make_node("Gather", ["embed_weight","input_ids"], [EMB_OUT], axis=0)],
    "embeds",
    [helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch","seq"])],
    [helper.make_tensor_value_info(EMB_OUT, TensorProto.FLOAT, ["batch","seq",H])],
    [init("embed_weight", emb_w)],
)
save(helper.make_model(g, opset_imports=[helper.make_opsetid("",17)]), f"{OUT}/embeds.onnx")

# ---- transformer.onnx (single layer causal attention with KV cache) ----
Wq=(np.random.randn(H,H)*0.2); Wk=(np.random.randn(H,H)*0.2)
Wv=(np.random.randn(H,H)*0.2); Wo=(np.random.randn(H,H)*0.2)
nodes=[]
C=[]  # constants/initializers
def const_i64(name, vals):
    C.append(nh.from_array(np.array(vals, dtype=np.int64), name=name))
const_i64("shape_split", [0,0,NH,HS])     # [b,s,nh,hs]
const_i64("shape_merge", [0,0,H])         # [b,s,H]
const_i64("perm_idx2", [2])
const_i64("idx2", [2])
C.append(nh.from_array(np.array(1.0/np.sqrt(HS), dtype=np.float32), name="scale"))
C.append(nh.from_array(np.array(1e9, dtype=np.float32), name="bignum"))
C.append(nh.from_array(np.array(1.0, dtype=np.float32), name="one_f"))
for nm,arr in [("Wq",Wq),("Wk",Wk),("Wv",Wv),("Wo",Wo)]:
    C.append(init(nm, arr))

def proj(name, w, out):
    nodes.append(helper.make_node("MatMul", ["emb", w], [name+"_lin"]))
    nodes.append(helper.make_node("Reshape", [name+"_lin","shape_split"], [name+"_r"]))
    nodes.append(helper.make_node("Transpose", [name+"_r"], [out], perm=[0,2,1,3]))

proj("q","Wq","q")    # [b,nh,s,hs]
proj("k","Wk","k_cur")
proj("v","Wv","v_cur")
# concat past
nodes.append(helper.make_node("Concat", ["past_key_values.0.key","k_cur"], ["present.0.key"], axis=2))
nodes.append(helper.make_node("Concat", ["past_key_values.0.value","v_cur"], ["present.0.value"], axis=2))
# scores
nodes.append(helper.make_node("Transpose", ["present.0.key"], ["k_t"], perm=[0,1,3,2]))  # [b,nh,hs,total]
nodes.append(helper.make_node("MatMul", ["q","k_t"], ["scores_raw"]))                     # [b,nh,s,total]
nodes.append(helper.make_node("Mul", ["scores_raw","scale"], ["scores_s"]))
# causal mask: k_off = total - s
nodes.append(helper.make_node("Shape", ["present.0.key"], ["pk_shape"]))
nodes.append(helper.make_node("Gather", ["pk_shape","idx2"], ["total_1"], axis=0))  # [1]
nodes.append(helper.make_node("Shape", ["q"], ["q_shape"]))
nodes.append(helper.make_node("Gather", ["q_shape","idx2"], ["s_1"], axis=0))       # [1]
nodes.append(helper.make_node("Concat", ["s_1","total_1"], ["st_shape"], axis=0))   # [s,total]
nodes.append(helper.make_node("ConstantOfShape", ["st_shape"], ["ones_st"],
             value=nh.from_array(np.array([1.0],dtype=np.float32), name="cof")))
nodes.append(helper.make_node("Sub", ["total_1","s_1"], ["koff_1"]))                # [1]
nodes.append(helper.make_node("Squeeze", ["koff_1"], ["koff"]))                     # scalar
nodes.append(helper.make_node("Trilu", ["ones_st","koff"], ["tri"], upper=0))       # keep col-row<=koff
nodes.append(helper.make_node("Sub", ["tri","one_f"], ["tri_m1"]))                  # kept->0, masked->-1
nodes.append(helper.make_node("Mul", ["tri_m1","bignum"], ["causal_add"]))          # masked->-1e9
nodes.append(helper.make_node("Add", ["scores_s","causal_add"], ["scores_c"]))
# attention_mask additive: (am-1)*1e9, reshape [b,1,1,total]
nodes.append(helper.make_node("Cast", ["attention_mask"], ["am_f"], to=TensorProto.FLOAT))
const_i64("am_shape", [0,1,1,-1])
nodes.append(helper.make_node("Reshape", ["am_f","am_shape"], ["am_r"]))
nodes.append(helper.make_node("Sub", ["am_r","one_f"], ["am_m1"]))
nodes.append(helper.make_node("Mul", ["am_m1","bignum"], ["am_add"]))
nodes.append(helper.make_node("Add", ["scores_c","am_add"], ["scores_f"]))
nodes.append(helper.make_node("Softmax", ["scores_f"], ["probs"], axis=-1))
nodes.append(helper.make_node("MatMul", ["probs","present.0.value"], ["ctx"]))      # [b,nh,s,hs]
nodes.append(helper.make_node("Transpose", ["ctx"], ["ctx_t"], perm=[0,2,1,3]))     # [b,s,nh,hs]
nodes.append(helper.make_node("Reshape", ["ctx_t","shape_merge"], ["ctx_m"]))       # [b,s,H]
nodes.append(helper.make_node("MatMul", ["ctx_m","Wo"], ["hidden_states"]))

ins=[
 helper.make_tensor_value_info("emb", TensorProto.FLOAT, ["batch","seq",H]),
 helper.make_tensor_value_info("attention_mask", TensorProto.INT32, ["batch","total"]),
 helper.make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT, ["batch",NH,"past",HS]),
 helper.make_tensor_value_info("past_key_values.0.value", TensorProto.FLOAT, ["batch",NH,"past",HS]),
]
# rename emb input to the embeddings output name expected by config
ins[0].name = EMB_OUT
for n in nodes:
    n.input[:] = [EMB_OUT if x=="emb" else x for x in n.input]
outs=[
 helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, ["batch","seq",H]),
 helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, ["batch",NH,"total",HS]),
 helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, ["batch",NH,"total",HS]),
]
g=helper.make_graph(nodes,"transformer",ins,outs,C)
save(helper.make_model(g, opset_imports=[helper.make_opsetid("",17)]), f"{OUT}/transformer.onnx")

# ---- lm_head.onnx ----
Wlm=(np.random.randn(H,V)*0.2).astype(np.float32)
g=helper.make_graph(
    [helper.make_node("MatMul", ["hidden_states","lm_weight"], ["logits"])],
    "lm_head",
    [helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, ["batch","seq",H])],
    [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch","seq",V])],
    [init("lm_weight", Wlm)],
)
save(helper.make_model(g, opset_imports=[helper.make_opsetid("",17)]), f"{OUT}/lm_head.onnx")
print("OK generated tiny pipeline onnx")
