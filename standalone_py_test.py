import onnxruntime_genai as og
from PIL import Image
import numpy as np

m = og.Model('C:\\Users\\yangselena\\onnxruntime-genai\\onnxruntime-genai\\test\\test_models\\sd')
p = og.ImageGeneratorParams(m)
p.set_prompt('a photo of a cat')

t = og.generate_image(m, p)

t_np = t.as_numpy()
#t_np = np.asarray(t)

#print(t_np.shape)
#print(t_np.dtype)
print(type(t_np))


#t_pil = Image.fromarray(t_np.astype(np.uint8))
t_pil = [Image.fromarray(t_np[i]) for i in range(t_np.shape[0])]

for i in range(len(t_pil)):
    t_pil[i].save(f'test_image_py_{i}.png')
