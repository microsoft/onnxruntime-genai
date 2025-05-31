// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "diffusion_model.h"
#include <vector>
#include <random>

namespace Generators {

// Constructor for StableDiffusion_Model
DiffusionModel::DiffusionModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model(std::move(config))
    , cpu_allocator_{Ort::Allocator::GetWithDefaultOptions()} {
  auto path = config_->config_path;
  auto text_encoder_path = path / "text_encoder" / "model.onnx";
  auto vae_model_path = path / "vae_decoder" / "model.onnx";
  auto unet_model_path = path / "unet" / "model.onnx";
  // Initialize encoder, deniosing(unet) and decoder sessions
  p_memory_info_ = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  std::unique_ptr<OrtSessionOptions> session_options = OrtSessionOptions::Create();

  p_session_ = OrtSession::Create(
      ort_env,
      text_encoder_path.c_str(),
      session_options.get());
  text_encoder_allocator_ = Ort::Allocator::Create(*p_session_, *p_memory_info_);

  auto unet_session_options = OrtSessionOptions::Create();
  p_unet_session_ = OrtSession::Create(
    ort_env,
    unet_model_path.c_str(),
    unet_session_options.get());
  unet_allocator_ = Ort::Allocator::Create(*p_unet_session_, *p_memory_info_);

  auto vae_session_options = OrtSessionOptions::Create();
  p_vae_session_ = OrtSession::Create(
      ort_env,
      vae_model_path.c_str(),
      vae_session_options.get());
  vae_allocator_ = Ort::Allocator::Create(*p_vae_session_, *p_memory_info_);

}

// CreateState method for Diffusion_Model
std::unique_ptr<State> DiffusionModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<DiffusionState>(*this, params);
}

// Generate method for Diffusion_Model
std::unique_ptr<OrtValue> DiffusionModel::Generate(const ImageGeneratorParams* params) const {
  // STEP.0 - Check inputs

  if (!params) {
    throw std::runtime_error("params is null");
  }

  if (params->prompts.size() != batch_size_) {
    throw std::runtime_error("Currently only support batch size == 1");
  }

  const std::string& prompt = params->prompts[0];
  if (prompt.empty()) {
    throw std::runtime_error("prompt is empty");
  }
  const std::string& negative_prompt = params->negative_prompts[0];
  if (!negative_prompt.empty()) {
    throw std::runtime_error("Currently negative prompt is not supported");
  }

  auto tokenizer = std::make_shared<Generators::Tokenizer>(*config_);

  auto input_ids = tokenizer->Encode(prompt.c_str());


  int32_t* sequences_data = &input_ids[0];
  size_t sequences_size = input_ids.size();


  // create the OrtSession
  //Ort::InitApi();
  //std::unique_ptr<OrtEnv> p_env = OrtEnv::Create(ORT_LOGGING_LEVEL_WARNING, "test");


  // enable_cuda_graph is false in prototype version
  // create input_ids tensor
  std::vector<int64_t> input_ids_shape{batch_size_, max_sequence_length};

  std::unique_ptr<OrtValue> p_input_tensor = OrtValue::CreateTensor<int32_t>(*text_encoder_allocator_, std::span{input_ids_shape});
  int32_t* input_ids_data = p_input_tensor->GetTensorMutableData<int32_t>();

  // if the length of input_ids is larger than max_sequence_length, we need to truncate it
  if (sequences_size > max_sequence_length) {
    std::copy(sequences_data, sequences_data + max_sequence_length, input_ids_data);
  }

  std::copy(sequences_data, sequences_data + sequences_size, input_ids_data);

  // if the length of input_ids is smaller than max_sequence_length, we need to pad it
  if (sequences_size < max_sequence_length) {
    std::fill(input_ids_data + sequences_size, input_ids_data + max_sequence_length, 0);
  }
  // Bind input tensors and run inference
  auto io_binding = OrtIoBinding::Create(*p_session_);
  io_binding->BindInput("input_ids", *p_input_tensor);

  // Bind output text_embeddings tensor

  std::vector<int64_t> output_embeddings_shape{batch_size_, max_sequence_length, hidden_size};
  std::unique_ptr<OrtValue> p_output_tensor = OrtValue::CreateTensor<float>(*text_encoder_allocator_, std::span{output_embeddings_shape});
  io_binding->BindOutput("text_embeddings", *p_output_tensor);

  std::unique_ptr<OrtRunOptions> run_options = OrtRunOptions::Create();

  io_binding->SynchronizeInputs();
  p_session_->Run(run_options.get(), *io_binding);
  io_binding->SynchronizeOutputs();

  // Get output text_embeddings tensor
  auto text_embeddings_output = io_binding->GetOutputValues();
  auto text_embeddings_tensor = text_embeddings_output[0].get();

  // auto text_embeddings_tensor_data = text_embeddings_tensor->GetTensorMutableData<float>();

  // Create a new OrtValue for the latents tensor
  int64_t latent_height = image_height / 8;
  int64_t latent_width = image_width / 8;
  //std::vector<int64_t> latents_shape{batch_size_, unet_channels_, latent_height, latent_width};
  // latents = torch.randn(latents_shape, device=self.device, dtype=latents_dtype, generator=self.generator)
  //  Create the tensor of latentss

  //std::unique_ptr<OrtValue> latents_tensor = OrtValue::CreateTensor<float>(*text_encoder_allocator_, std::span{latents_shape});
  std::unique_ptr<OrtValue> latents_tensor = CreateLatents(latent_height, latent_width, text_encoder_allocator_.get());
  float* latents_data = latents_tensor->GetTensorMutableData<float>();

  // Create a random number generator and normal distribution
  //std::random_device rd;
  //std::mt19937 gen(rd());
  //std::normal_distribution<float> dist(0.0, 1.0);

  // Fill the latents_tensor_data with random values
  // size_t step_index = 0;
  for (int i = 0; i < batch_size_ * unet_channels_ * latent_height * latent_width; i++) {
    // Scale the initial noise by the standard deviation required by the scheduler
    latents_data[i] = latents_data[i] * init_noise_sigma;
  }

  // Denoising latents
  std::vector<int64_t> tiemsteps = {999, 759, 519, 279};
  // beta_start**0.5, beta_end**0.5, torch.linspace()
  std::vector<float> alphas_cumprod = {0.9991f, 0.9983f, 0.9974f, 0.9966f, 0.9957f, 0.9948f, 0.9940f, 0.9931f, 0.9922f,
                                       0.9913f, 0.9904f, 0.9895f, 0.9886f, 0.9877f, 0.9868f, 0.9859f, 0.9850f, 0.9841f,
                                       0.9832f, 0.9822f, 0.9813f, 0.9804f, 0.9794f, 0.9785f, 0.9776f, 0.9766f, 0.9757f,
                                       0.9747f, 0.9737f, 0.9728f, 0.9718f, 0.9708f, 0.9698f, 0.9689f, 0.9679f, 0.9669f,
                                       0.9659f, 0.9649f, 0.9639f, 0.9629f, 0.9619f, 0.9609f, 0.9599f, 0.9588f, 0.9578f,
                                       0.9568f, 0.9557f, 0.9547f, 0.9537f, 0.9526f, 0.9516f, 0.9505f, 0.9495f, 0.9484f,
                                       0.9473f, 0.9463f, 0.9452f, 0.9441f, 0.9430f, 0.9420f, 0.9409f, 0.9398f, 0.9387f,
                                       0.9376f, 0.9365f, 0.9354f, 0.9343f, 0.9332f, 0.9320f, 0.9309f, 0.9298f, 0.9287f,
                                       0.9275f, 0.9264f, 0.9252f, 0.9241f, 0.9229f, 0.9218f, 0.9206f, 0.9195f, 0.9183f,
                                       0.9171f, 0.9160f, 0.9148f, 0.9136f, 0.9124f, 0.9112f, 0.9100f, 0.9089f, 0.9077f,
                                       0.9065f, 0.9052f, 0.9040f, 0.9028f, 0.9016f, 0.9004f, 0.8992f, 0.8979f, 0.8967f,
                                       0.8955f, 0.8942f, 0.8930f, 0.8917f, 0.8905f, 0.8892f, 0.8880f, 0.8867f, 0.8854f,
                                       0.8842f, 0.8829f, 0.8816f, 0.8804f, 0.8791f, 0.8778f, 0.8765f, 0.8752f, 0.8739f,
                                       0.8726f, 0.8713f, 0.8700f, 0.8687f, 0.8674f, 0.8661f, 0.8647f, 0.8634f, 0.8621f,
                                       0.8607f, 0.8594f, 0.8581f, 0.8567f, 0.8554f, 0.8540f, 0.8527f, 0.8513f, 0.8500f,
                                       0.8486f, 0.8473f, 0.8459f, 0.8445f, 0.8431f, 0.8418f, 0.8404f, 0.8390f, 0.8376f,
                                       0.8362f, 0.8348f, 0.8334f, 0.8320f, 0.8306f, 0.8292f, 0.8278f, 0.8264f, 0.8250f,
                                       0.8236f, 0.8221f, 0.8207f, 0.8193f, 0.8179f, 0.8164f, 0.8150f, 0.8136f, 0.8121f,
                                       0.8107f, 0.8092f, 0.8078f, 0.8063f, 0.8049f, 0.8034f, 0.8019f, 0.8005f, 0.7990f,
                                       0.7975f, 0.7960f, 0.7946f, 0.7931f, 0.7916f, 0.7901f, 0.7886f, 0.7871f, 0.7856f,
                                       0.7842f, 0.7827f, 0.7812f, 0.7796f, 0.7781f, 0.7766f, 0.7751f, 0.7736f, 0.7721f,
                                       0.7706f, 0.7690f, 0.7675f, 0.7660f, 0.7645f, 0.7629f, 0.7614f, 0.7599f, 0.7583f,
                                       0.7568f, 0.7552f, 0.7537f, 0.7521f, 0.7506f, 0.7490f, 0.7475f, 0.7459f, 0.7444f,
                                       0.7428f, 0.7412f, 0.7397f, 0.7381f, 0.7365f, 0.7350f, 0.7334f, 0.7318f, 0.7302f,
                                       0.7286f, 0.7271f, 0.7255f, 0.7239f, 0.7223f, 0.7207f, 0.7191f, 0.7175f, 0.7159f,
                                       0.7143f, 0.7127f, 0.7111f, 0.7095f, 0.7079f, 0.7063f, 0.7047f, 0.7031f, 0.7015f,
                                       0.6999f, 0.6982f, 0.6966f, 0.6950f, 0.6934f, 0.6918f, 0.6901f, 0.6885f, 0.6869f,
                                       0.6852f, 0.6836f, 0.6820f, 0.6803f, 0.6787f, 0.6771f, 0.6754f, 0.6738f, 0.6722f,
                                       0.6705f, 0.6689f, 0.6672f, 0.6656f, 0.6639f, 0.6623f, 0.6606f, 0.6590f, 0.6573f,
                                       0.6557f, 0.6540f, 0.6524f, 0.6507f, 0.6490f, 0.6474f, 0.6457f, 0.6441f, 0.6424f,
                                       0.6407f, 0.6391f, 0.6374f, 0.6357f, 0.6341f, 0.6324f, 0.6307f, 0.6291f, 0.6274f,
                                       0.6257f, 0.6241f, 0.6224f, 0.6207f, 0.6190f, 0.6174f, 0.6157f, 0.6140f, 0.6123f,
                                       0.6107f, 0.6090f, 0.6073f, 0.6056f, 0.6039f, 0.6023f, 0.6006f, 0.5989f, 0.5972f,
                                       0.5955f, 0.5939f, 0.5922f, 0.5905f, 0.5888f, 0.5871f, 0.5855f, 0.5838f, 0.5821f,
                                       0.5804f, 0.5787f, 0.5770f, 0.5754f, 0.5737f, 0.5720f, 0.5703f, 0.5686f, 0.5669f,
                                       0.5652f, 0.5636f, 0.5619f, 0.5602f, 0.5585f, 0.5568f, 0.5551f, 0.5535f, 0.5518f,
                                       0.5501f, 0.5484f, 0.5467f, 0.5450f, 0.5434f, 0.5417f, 0.5400f, 0.5383f, 0.5366f,
                                       0.5350f, 0.5333f, 0.5316f, 0.5299f, 0.5282f, 0.5266f, 0.5249f, 0.5232f, 0.5215f,
                                       0.5199f, 0.5182f, 0.5165f, 0.5148f, 0.5132f, 0.5115f, 0.5098f, 0.5082f, 0.5065f,
                                       0.5048f, 0.5032f, 0.5015f, 0.4998f, 0.4982f, 0.4965f, 0.4948f, 0.4932f, 0.4915f,
                                       0.4898f, 0.4882f, 0.4865f, 0.4849f, 0.4832f, 0.4816f, 0.4799f, 0.4782f, 0.4766f,
                                       0.4749f, 0.4733f, 0.4716f, 0.4700f, 0.4684f, 0.4667f, 0.4651f, 0.4634f, 0.4618f,
                                       0.4601f, 0.4585f, 0.4569f, 0.4552f, 0.4536f, 0.4520f, 0.4503f, 0.4487f, 0.4471f,
                                       0.4455f, 0.4438f, 0.4422f, 0.4406f, 0.4390f, 0.4374f, 0.4357f, 0.4341f, 0.4325f,
                                       0.4309f, 0.4293f, 0.4277f, 0.4261f, 0.4245f, 0.4229f, 0.4213f, 0.4197f, 0.4181f,
                                       0.4165f, 0.4149f, 0.4133f, 0.4117f, 0.4101f, 0.4086f, 0.4070f, 0.4054f, 0.4038f,
                                       0.4022f, 0.4007f, 0.3991f, 0.3975f, 0.3960f, 0.3944f, 0.3928f, 0.3913f, 0.3897f,
                                       0.3882f, 0.3866f, 0.3850f, 0.3835f, 0.3819f, 0.3804f, 0.3789f, 0.3773f, 0.3758f,
                                       0.3742f, 0.3727f, 0.3712f, 0.3697f, 0.3681f, 0.3666f, 0.3651f, 0.3636f, 0.3621f,
                                       0.3605f, 0.3590f, 0.3575f, 0.3560f, 0.3545f, 0.3530f, 0.3515f, 0.3500f, 0.3485f,
                                       0.3470f, 0.3456f, 0.3441f, 0.3426f, 0.3411f, 0.3396f, 0.3382f, 0.3367f, 0.3352f,
                                       0.3338f, 0.3323f, 0.3308f, 0.3294f, 0.3279f, 0.3265f, 0.3250f, 0.3236f, 0.3222f,
                                       0.3207f, 0.3193f, 0.3178f, 0.3164f, 0.3150f, 0.3136f, 0.3122f, 0.3107f, 0.3093f,
                                       0.3079f, 0.3065f, 0.3051f, 0.3037f, 0.3023f, 0.3009f, 0.2995f, 0.2981f, 0.2967f,
                                       0.2953f, 0.2940f, 0.2926f, 0.2912f, 0.2899f, 0.2885f, 0.2871f, 0.2858f, 0.2844f,
                                       0.2831f, 0.2817f, 0.2804f, 0.2790f, 0.2777f, 0.2763f, 0.2750f, 0.2737f, 0.2723f,
                                       0.2710f, 0.2697f, 0.2684f, 0.2671f, 0.2658f, 0.2645f, 0.2631f, 0.2618f, 0.2606f,
                                       0.2593f, 0.2580f, 0.2567f, 0.2554f, 0.2541f, 0.2528f, 0.2516f, 0.2503f, 0.2490f,
                                       0.2478f, 0.2465f, 0.2453f, 0.2440f, 0.2428f, 0.2415f, 0.2403f, 0.2391f, 0.2378f,
                                       0.2366f, 0.2354f, 0.2341f, 0.2329f, 0.2317f, 0.2305f, 0.2293f, 0.2281f, 0.2269f,
                                       0.2257f, 0.2245f, 0.2233f, 0.2221f, 0.2209f, 0.2198f, 0.2186f, 0.2174f, 0.2163f,
                                       0.2151f, 0.2139f, 0.2128f, 0.2116f, 0.2105f, 0.2093f, 0.2082f, 0.2071f, 0.2059f,
                                       0.2048f, 0.2037f, 0.2026f, 0.2014f, 0.2003f, 0.1992f, 0.1981f, 0.1970f, 0.1959f,
                                       0.1948f, 0.1937f, 0.1926f, 0.1915f, 0.1905f, 0.1894f, 0.1883f, 0.1872f, 0.1862f,
                                       0.1851f, 0.1841f, 0.1830f, 0.1820f, 0.1809f, 0.1799f, 0.1788f, 0.1778f, 0.1768f,
                                       0.1757f, 0.1747f, 0.1737f, 0.1727f, 0.1717f, 0.1707f, 0.1696f, 0.1686f, 0.1677f,
                                       0.1667f, 0.1657f, 0.1647f, 0.1637f, 0.1627f, 0.1618f, 0.1608f, 0.1598f, 0.1589f,
                                       0.1579f, 0.1569f, 0.1560f, 0.1550f, 0.1541f, 0.1532f, 0.1522f, 0.1513f, 0.1504f,
                                       0.1494f, 0.1485f, 0.1476f, 0.1467f, 0.1458f, 0.1449f, 0.1440f, 0.1431f, 0.1422f,
                                       0.1413f, 0.1404f, 0.1395f, 0.1386f, 0.1378f, 0.1369f, 0.1360f, 0.1352f, 0.1343f,
                                       0.1334f, 0.1326f, 0.1317f, 0.1309f, 0.1301f, 0.1292f, 0.1284f, 0.1276f, 0.1267f,
                                       0.1259f, 0.1251f, 0.1243f, 0.1235f, 0.1227f, 0.1219f, 0.1211f, 0.1203f, 0.1195f,
                                       0.1187f, 0.1179f, 0.1171f, 0.1163f, 0.1155f, 0.1148f, 0.1140f, 0.1132f, 0.1125f,
                                       0.1117f, 0.1110f, 0.1102f, 0.1095f, 0.1087f, 0.1080f, 0.1073f, 0.1065f, 0.1058f,
                                       0.1051f, 0.1044f, 0.1036f, 0.1029f, 0.1022f, 0.1015f, 0.1008f, 0.1001f, 0.0994f,
                                       0.0987f, 0.0980f, 0.0973f, 0.0967f, 0.0960f, 0.0953f, 0.0946f, 0.0940f, 0.0933f,
                                       0.0926f, 0.0920f, 0.0913f, 0.0907f, 0.0900f, 0.0894f, 0.0887f, 0.0881f, 0.0875f,
                                       0.0868f, 0.0862f, 0.0856f, 0.0850f, 0.0844f, 0.0837f, 0.0831f, 0.0825f, 0.0819f,
                                       0.0813f, 0.0807f, 0.0801f, 0.0795f, 0.0789f, 0.0784f, 0.0778f, 0.0772f, 0.0766f,
                                       0.0761f, 0.0755f, 0.0749f, 0.0744f, 0.0738f, 0.0732f, 0.0727f, 0.0721f, 0.0716f,
                                       0.0711f, 0.0705f, 0.0700f, 0.0694f, 0.0689f, 0.0684f, 0.0679f, 0.0673f, 0.0668f,
                                       0.0663f, 0.0658f, 0.0653f, 0.0648f, 0.0643f, 0.0638f, 0.0633f, 0.0628f, 0.0623f,
                                       0.0618f, 0.0613f, 0.0608f, 0.0604f, 0.0599f, 0.0594f, 0.0589f, 0.0585f, 0.0580f,
                                       0.0575f, 0.0571f, 0.0566f, 0.0562f, 0.0557f, 0.0553f, 0.0548f, 0.0544f, 0.0539f,
                                       0.0535f, 0.0531f, 0.0526f, 0.0522f, 0.0518f, 0.0514f, 0.0509f, 0.0505f, 0.0501f,
                                       0.0497f, 0.0493f, 0.0489f, 0.0485f, 0.0481f, 0.0477f, 0.0473f, 0.0469f, 0.0465f,
                                       0.0461f, 0.0457f, 0.0453f, 0.0450f, 0.0446f, 0.0442f, 0.0438f, 0.0435f, 0.0431f,
                                       0.0427f, 0.0424f, 0.0420f, 0.0416f, 0.0413f, 0.0409f, 0.0406f, 0.0402f, 0.0399f,
                                       0.0395f, 0.0392f, 0.0389f, 0.0385f, 0.0382f, 0.0379f, 0.0375f, 0.0372f, 0.0369f,
                                       0.0365f, 0.0362f, 0.0359f, 0.0356f, 0.0353f, 0.0350f, 0.0347f, 0.0343f, 0.0340f,
                                       0.0337f, 0.0334f, 0.0331f, 0.0328f, 0.0325f, 0.0323f, 0.0320f, 0.0317f, 0.0314f,
                                       0.0311f, 0.0308f, 0.0305f, 0.0303f, 0.0300f, 0.0297f, 0.0295f, 0.0292f, 0.0289f,
                                       0.0286f, 0.0284f, 0.0281f, 0.0279f, 0.0276f, 0.0274f, 0.0271f, 0.0268f, 0.0266f,
                                       0.0264f, 0.0261f, 0.0259f, 0.0256f, 0.0254f, 0.0251f, 0.0249f, 0.0247f, 0.0244f,
                                       0.0242f, 0.0240f, 0.0237f, 0.0235f, 0.0233f, 0.0231f, 0.0229f, 0.0226f, 0.0224f,
                                       0.0222f, 0.0220f, 0.0218f, 0.0216f, 0.0214f, 0.0212f, 0.0210f, 0.0207f, 0.0205f,
                                       0.0203f, 0.0201f, 0.0200f, 0.0198f, 0.0196f, 0.0194f, 0.0192f, 0.0190f, 0.0188f,
                                       0.0186f, 0.0184f, 0.0182f, 0.0181f, 0.0179f, 0.0177f, 0.0175f, 0.0174f, 0.0172f,
                                       0.0170f, 0.0168f, 0.0167f, 0.0165f, 0.0163f, 0.0162f, 0.0160f, 0.0158f, 0.0157f,
                                       0.0155f, 0.0154f, 0.0152f, 0.0151f, 0.0149f, 0.0147f, 0.0146f, 0.0144f, 0.0143f,
                                       0.0142f, 0.0140f, 0.0139f, 0.0137f, 0.0136f, 0.0134f, 0.0133f, 0.0132f, 0.0130f,
                                       0.0129f, 0.0127f, 0.0126f, 0.0125f, 0.0123f, 0.0122f, 0.0121f, 0.0120f, 0.0118f,
                                       0.0117f, 0.0116f, 0.0115f, 0.0113f, 0.0112f, 0.0111f, 0.0110f, 0.0109f, 0.0107f,
                                       0.0106f, 0.0105f, 0.0104f, 0.0103f, 0.0102f, 0.0101f, 0.0100f, 0.0098f, 0.0097f,
                                       0.0096f, 0.0095f, 0.0094f, 0.0093f, 0.0092f, 0.0091f, 0.0090f, 0.0089f, 0.0088f,
                                       0.0087f, 0.0086f, 0.0085f, 0.0084f, 0.0083f, 0.0082f, 0.0082f, 0.0081f, 0.0080f,
                                       0.0079f, 0.0078f, 0.0077f, 0.0076f, 0.0075f, 0.0074f, 0.0074f, 0.0073f, 0.0072f,
                                       0.0071f, 0.0070f, 0.0070f, 0.0069f, 0.0068f, 0.0067f, 0.0066f, 0.0066f, 0.0065f,
                                       0.0064f, 0.0063f, 0.0063f, 0.0062f, 0.0061f, 0.0061f, 0.0060f, 0.0059f, 0.0058f,
                                       0.0058f, 0.0057f, 0.0056f, 0.0056f, 0.0055f, 0.0054f, 0.0054f, 0.0053f, 0.0053f,
                                       0.0052f, 0.0051f, 0.0051f, 0.0050f, 0.0049f, 0.0049f, 0.0048f, 0.0048f, 0.0047f,
                                       0.0047f};

  // create the unet session
  auto io_binding_unet = OrtIoBinding::Create(*p_unet_session_);

  for (size_t i = 0; i < tiemsteps.size(); i++) {
    std::vector<int64_t> timestep_shape{1};
    std::vector<float> timestep_data{static_cast<float>(tiemsteps[i])};
    std::unique_ptr<OrtValue> timestep_tensor = OrtValue::CreateTensor<float>(*p_memory_info_, std::span{timestep_data}, std::span{timestep_shape});

    // Create the input params map for the denoising step
    std::map<std::string, OrtValue*> input_params;
    input_params["sample"] = latents_tensor.get();
    input_params["timestep"] = timestep_tensor.get();
    input_params["encoder_hidden_states"] = text_embeddings_tensor;

    // Bind the input tensors and run inference


    io_binding_unet->BindInput("sample", *input_params["sample"]);
    io_binding_unet->BindInput("timestep", *input_params["timestep"]);
    io_binding_unet->BindInput("encoder_hidden_states", *input_params["encoder_hidden_states"]);

    // Bind the output latent tensor
    std::vector<int64_t> output_latent_shape{batch_size_, unet_channels_, latent_height, latent_width};
    std::unique_ptr<OrtValue> output_latent_tensor = OrtValue::CreateTensor<float>(*unet_allocator_, std::span{output_latent_shape});
    io_binding_unet->BindOutput("latent", *output_latent_tensor);

    // Run the unet model

    io_binding_unet->SynchronizeInputs();
    p_unet_session_->Run(run_options.get(), *io_binding_unet);
    io_binding_unet->SynchronizeOutputs();

    // Get the output latent tensor
    auto unet_output_tensor = io_binding_unet->GetOutputValues();
    auto noise_pred_data = unet_output_tensor[0]->GetTensorMutableData<float>();

    // Post-process noise_pred to step it
    // 1. get previous step value
    size_t prev_step_index = i + 1;
    int64_t prev_timestep = 0;
    if (prev_step_index < tiemsteps.size()) {
      prev_timestep = tiemsteps[prev_step_index];
    } else {
      prev_timestep = tiemsteps[i];
    }

    // 2. compute alphas, betas
    auto alpha_prod_t = alphas_cumprod[tiemsteps[i]];
    auto alpha_prod_t_prev = prev_timestep >= 0 ? alphas_cumprod[prev_timestep] : alphas_cumprod[0];

    auto beta_prod_t = 1 - alpha_prod_t;
    auto beta_prod_t_prev = 1 - alpha_prod_t_prev;

    // 3. Get scalings for boundary conditions
    float sigma_data = 0.5;  // Default: 0.5
    float timestep_scaling = 10.0;
    float scaled_timestep = tiemsteps[i] * timestep_scaling;

    float c_skip = (sigma_data * sigma_data) / (scaled_timestep * scaled_timestep + sigma_data * sigma_data);
    float c_out = scaled_timestep / std::sqrt(scaled_timestep * scaled_timestep + sigma_data * sigma_data);

    // 4. Compute the predicted original sample x_0 based on the model parameterization
    // prediction_type == "epsilon":  # noise-prediction
    // predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    std::vector<int64_t> predicted_sample_shape{batch_size_, unet_channels_, latent_height, latent_width};
    std::vector<float> predicted_sample_data(batch_size_ * unet_channels_ * latent_height * latent_width);
    for (int j = 0; j < predicted_sample_data.size(); j++) {
      predicted_sample_data[j] = (latents_data[j] - std::sqrt(beta_prod_t) * noise_pred_data[j]) / std::sqrt(alpha_prod_t);
    }
    std::unique_ptr<OrtValue> predicted_sample_tensor = OrtValue::CreateTensor<float>(*p_memory_info_, std::span{predicted_sample_data}, std::span{predicted_sample_shape});

    // 5. Clip or threshold "predicted x_0"
    // In SD turbo, this step is skipped

    // 6. Denoise model output using boundary conditions
    // denoised = c_out * predicted_original_sample + c_skip * sample
    for (int j = 0; j < predicted_sample_data.size(); j++) {
      predicted_sample_data[j] = c_out * predicted_sample_data[j] + c_skip * latents_data[j];
    }

    // 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
    // Noise is not used on the final timestep of the timestep schedule.
    // This also means that noise is not used for one-step sampling.
    //  prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
    if (i != tiemsteps.size() - 1) {
      auto noise = CreateLatents(latent_height, latent_width, unet_allocator_.get());
      auto noise_data = noise->GetTensorMutableData<float>();
      for (int j = 0; j < predicted_sample_data.size(); j++) {
        latents_data[j] = std::sqrt(alpha_prod_t_prev) * predicted_sample_data[j] + std::sqrt(beta_prod_t_prev) * noise_data[j];
      }
    } else {
      for (int j = 0; j < predicted_sample_data.size(); j++) {
        latents_data[j] = predicted_sample_data[j] / vae_scaling_factor;
      }
    }
  }

  // VAE decode latents
  // Bind the vae input tensors
  auto io_binding_vae = OrtIoBinding::Create(*p_vae_session_);

  io_binding_vae->BindInput("latent", *latents_tensor);

  // Bind the output latent tensor
  std::vector<int64_t> output_image_shape{batch_size_, 3, image_height, image_width};
  std::unique_ptr<OrtValue> output_image_tensor = OrtValue::CreateTensor<float>(*vae_allocator_, std::span{output_image_shape});
  // auto x = output_image_tensor->GetTensorMutableData<float>();

  io_binding_vae->BindOutput("images", *output_image_tensor);

  // Run the vae decoder model

  io_binding_vae->SynchronizeInputs();
  p_vae_session_->Run(run_options.get(), *io_binding_vae);
  io_binding_vae->SynchronizeOutputs();

  // Get the output image tensor
  auto vae_output_tensor = io_binding_vae->GetOutputValues();

  // auto image_tensor_data = vae_output_tensor[0]->GetTensorMutableData<float>();

  auto images = std::move(vae_output_tensor[0]);

  std::vector<int64_t> post_processed_image_shape{static_cast<int64_t>(batch_size_), image_height, image_width, 3};
  auto post_processed_image_tensor = OrtValue::CreateTensor<uint8_t>(*vae_allocator_, post_processed_image_shape);


  // temporary code: leaked memory
  //uint8_t* image_data = new uint8_t[batch_size_ * image_height * image_width * 3];

  for (size_t B = 0; B < batch_size_; ++B) {
    for (size_t H = 0; H < image_height; ++H) {
      for (size_t W = 0; W < image_width; ++W) {
        for (size_t C = 0; C < 3; ++C) {
          size_t index_images = B * image_height * image_width * 3 + C * image_height * image_width + H * image_width + W;
          size_t index_post_processed = B * image_height * image_width * 3 + H * image_width * 3 + W * 3 + C;

          float image_value = images->GetTensorMutableData<float>()[index_images];
          uint8_t post_processed_value = static_cast<uint8_t>(std::clamp((image_value + 1.0f) * 255.0f / 2.0f, 0.f, 255.f));

          // if (C == 2) {
          //   std::cout << (int)post_processed_value << ", ";
          // }

          post_processed_image_tensor->GetTensorMutableData<uint8_t>()[index_post_processed] = post_processed_value;
        }
      }
    }
  }

  return post_processed_image_tensor;
}

// CreateLatents method for Diffusion_Model
std::unique_ptr<OrtValue> DiffusionModel::CreateLatents(int64_t latent_height, int64_t latent_width, Ort::Allocator* allocator) const {
  // Create the tensor of latentss
  std::vector<int64_t> latents_shape{batch_size_, unet_channels_, latent_height, latent_width};
  std::unique_ptr<OrtValue> latents_tensor = OrtValue::CreateTensor<float>(*allocator, std::span{latents_shape});
  float* latents_data = latents_tensor->GetTensorMutableData<float>();

  // Create a random number generator and normal distribution
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0, 1.0);

  // Fill the latents_tensor_data with random values

  for (int i = 0; i < batch_size_ * unet_channels_ * latent_height * latent_width; i++) {
    
    latents_data[i] = dist(gen);
  }
  return latents_tensor;
}


// Constructor for DiffusionState
DiffusionState::DiffusionState(const DiffusionModel& model, const GeneratorParams& params)
    : State(params, model), model_(model) {

}

// Run method for DiffusionState
DeviceSpan<float> DiffusionState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {

  return DeviceSpan<float>();
}


}  // namespace Generators
