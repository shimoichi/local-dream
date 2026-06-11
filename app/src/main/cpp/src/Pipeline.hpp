#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <MNN/Interpreter.hpp>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include "Config.hpp"
#include "DPMSolverMultistepScheduler.hpp"
#include "EulerAncestralDiscreteScheduler.hpp"
#include "EulerDiscreteScheduler.hpp"
#include "LCMScheduler.hpp"
#include "LaplacianBlend.hpp"
#include "Logger.hpp"
#include "MemUtils.hpp"
#include "MnnUtils.hpp"
#include "PromptCache.hpp"
#include "SDUtils.hpp"
#include "Scheduler.hpp"
#include "TextEncoder.hpp"
#include "Tiling.hpp"

// All per-request parameters. Image/mask buffers are pre-decoded by the
// request parser:
//   img_data:       [1,3,H,W] float, -1..1
//   mask_data:      [1,4,H/8,W/8] float, 0..1 (latent-space mask)
//   mask_data_full: [1,3,H,W] float, 0..1 (pixel-space mask)
struct GenerationRequest {
  std::string prompt;
  std::string negative_prompt;
  int steps = 20;
  float cfg = 7.5f;
  unsigned seed = 0;
  std::string scheduler_type = "dpm";
  bool use_opencl = false;
  bool show_diffusion_process = false;
  int show_diffusion_stride = 1;
  int width = 512;
  int height = 512;
  float denoise_strength = 0.6f;
  bool img2img = false;
  bool has_mask = false;
  std::vector<float> img_data;
  std::vector<float> mask_data;
  std::vector<float> mask_data_full;

  // Wire encoding for images sent back to the client: "raw" (RGB bytes),
  // "jpeg" or "png", each base64-wrapped inside the SSE JSON. Raw stays the
  // default for backward compatibility; jpeg shrinks a per-step preview by
  // an order of magnitude, which matters for remote (LAN) clients.
  std::string preview_format = "raw";
  std::string output_format = "raw";

  // SDXL aspect-ratio padded inpaint: when a non-1:1 ratio is requested for an
  // SDXL model, the pipeline still generates a 1024x1024 canvas, masks the
  // outer black border, and crops the centered region before returning.
  bool aspect_pad_inpaint = false;
  int target_crop_width = 0;
  int target_crop_height = 0;
  // True only when the base image is the synthetic white-on-black canvas
  // (txt2img path); this lets the VAE-encoder cache safely persist the encoded
  // latents per target size. False when the user uploaded their own image
  // (img2img / inpaint), where the base image is content-dependent.
  bool aspect_pad_synthetic_base = false;
  // True when the request actually carried a "mask" field (real inpaint).
  // False when the mask was auto-installed by the aspect-padding pipeline
  // (txt2img / img2img-with-aspect). Used to decide whether to laplacian-blend
  // the decoded image against the original after generation.
  bool user_supplied_mask = false;

  // Ultrafix: tiled img2img over an arbitrarily large (upscaled) input.
  // VAE encode, every UNet denoising step and the final decode all run as
  // overlapping fixed-size tiles blended back together, so the QNN graphs
  // keep their native size while width/height describe the full image.
  bool ultrafix = false;
  // UNet tile edge in pixel space = the model's native generation size
  // (client-provided for sd15npu, forced to 1024 for SDXL).
  int ultrafix_tile = 512;
};

// step / total_steps / optional base64 preview image.
using ProgressCallback = std::function<void(int, int, const std::string &)>;

// CLIP outputs for the [negative, positive] batch. `hidden` is what the UNet
// consumes as encoder_hidden_states; `pooled` and `time_ids` only exist for
// SDXL.
struct Conditioning {
  std::vector<float> hidden;    // [2, 77, hidden_dim]
  std::vector<float> pooled;    // [2, pooled_dim]
  std::vector<float> time_ids;  // [2, 6]
  int hidden_dim = 0;
  int pooled_dim = 0;

  float *negHidden() { return hidden.data(); }
  float *posHidden() { return hidden.data() + 77 * hidden_dim; }
  float *negPooled() { return pooled.empty() ? nullptr : pooled.data(); }
  float *posPooled() {
    return pooled.empty() ? nullptr : pooled.data() + pooled_dim;
  }
};

inline std::unique_ptr<Scheduler> createScheduler(
    const std::string &scheduler_type, const char *timestep_spacing) {
  if (scheduler_type == "euler_a" || scheduler_type == "eulera" ||
      scheduler_type == "euler_a_karras") {
    bool use_karras = (scheduler_type == "euler_a_karras");
    return std::make_unique<EulerAncestralDiscreteScheduler>(
        1000, 0.00085f, 0.012f, "scaled_linear", "epsilon", timestep_spacing, 0,
        false, use_karras);
  }
  if (scheduler_type == "euler" || scheduler_type == "euler_karras") {
    bool use_karras = (scheduler_type == "euler_karras");
    return std::make_unique<EulerDiscreteScheduler>(
        1000, 0.00085f, 0.012f, "scaled_linear", "epsilon", timestep_spacing, 0,
        false, use_karras);
  }
  if (scheduler_type == "lcm") {
    return std::make_unique<LCMScheduler>(1000, 0.00085f, 0.012f,
                                          "scaled_linear", "epsilon", 50, 10.0f,
                                          true, false);
  }
  if (scheduler_type == "dpm_sde" || scheduler_type == "dpm_sde_karras") {
    bool use_karras = (scheduler_type == "dpm_sde_karras");
    return std::make_unique<DPMSolverMultistepScheduler>(
        1000, 0.00085f, 0.012f, "scaled_linear", 2, "epsilon", timestep_spacing,
        use_karras, "sde-dpmsolver++");
  }
  // Default to DPM solver; "dpm_karras" enables Karras sigma schedule.
  bool use_karras = (scheduler_type == "dpm_karras");
  return std::make_unique<DPMSolverMultistepScheduler>(
      1000, 0.00085f, 0.012f, "scaled_linear", 2, "epsilon", timestep_spacing,
      use_karras);
}

// Base class for the three model formats (sd15cpu / sd15npu / sdxl). The
// whole generation algorithm lives in generate(); subclasses only provide the
// per-stage model execution (CLIP, VAE encode/decode, UNet step), lifecycle
// (persistent vs per-request vs lowram loading) and a few capability flags.
class Pipeline {
 public:
  Pipeline(TextEncoder &text_encoder, std::string model_dir, bool sdxl,
           bool use_v_pred)
      : text_encoder_(text_encoder),
        model_dir_(std::move(model_dir)),
        sdxl_(sdxl),
        use_v_pred_(use_v_pred) {}
  virtual ~Pipeline() = default;
  Pipeline(const Pipeline &) = delete;
  Pipeline &operator=(const Pipeline &) = delete;

  // Loads whatever the format keeps resident. Returns false on failure.
  virtual bool initialize() = 0;
  virtual bool supportsImg2Img() const = 0;
  bool isSdxl() const { return sdxl_; }
  // Ultrafix needs the VAE encoder (img2img) plus fixed-size graphs that can
  // be run as tiles; the MNN (CPU) format has neither constraint nor need.
  virtual bool supportsUltrafix() const {
    return vaeTilingSupported() && supportsImg2Img();
  }

  void setSafetyChecker(MNN::Interpreter *interpreter, MNN::Session *session,
                        float threshold) {
    safety_interpreter_ = interpreter;
    safety_session_ = session;
    nsfw_threshold_ = threshold;
  }

  // Mutates `req` only to release the decoded image buffer once it is no
  // longer needed (a ~190 MB allocation at ultrafix sizes).
  GenerationResult generate(GenerationRequest &req,
                            const ProgressCallback &progress_callback);

 protected:
  // --- stage hooks -------------------------------------------------------
  // Runs CLIP for the sides that missed the prompt cache, writing into the
  // matching halves of `cond`.
  virtual void encodeText(const ProcessedPromptPair &prompts,
                          bool need_negative, bool need_positive,
                          Conditioning &cond) = 0;
  // Single VAE-encoder pass at the current global IO dimensions.
  virtual void vaeEncode(const GenerationRequest &req, const float *image,
                         float *mean, float *std_dev) = 0;
  // Single VAE-decoder pass at the current global IO dimensions.
  virtual void vaeDecode(const GenerationRequest &req, const float *latents,
                         float *pixels) = 0;
  // Called before / after the denoising loop (load per-request UNet, lowram).
  virtual void beginDenoise(const GenerationRequest &req) {}
  virtual void endDenoise() {}
  // One UNet evaluation over the [negative, positive] batch. When
  // skip_uncond is true only the positive half of out_batch2 is written.
  virtual void runUnetStep(const GenerationRequest &req,
                           const float *latents_batch2, int timestep,
                           bool skip_uncond, Conditioning &cond,
                           float *out_batch2) = 0;

  // --- capabilities ------------------------------------------------------
  // With cfg = 1.0, noise_pred = uncond + 1*(txt - uncond) = txt, so the
  // unconditional pass is redundant. Backends that evaluate the two batch
  // halves separately (QNN) can skip it to halve UNet time; MNN runs both
  // in a single graph call so the optimization does not apply there.
  virtual bool canSkipUncond() const = 0;
  virtual bool previewSupported() const = 0;
  // Tiled VAE encode/decode for outputs above the fixed graph size (QNN
  // formats only: MNN handles large resolutions natively).
  virtual bool vaeTilingSupported() const { return false; }
  // Fixed VAE graph edge in pixel space (512 for SD1.5 NPU, 1024 for SDXL).
  virtual int vaeTilePixelSize() const { return 512; }
  // Catch-all invoked when generate() exits (normal return or exception);
  // lowram pipelines release any stage model still loaded.
  virtual void releaseTransientModels() {}

  float vaeScale() const { return sdxl_ ? 0.13025f : 0.18215f; }

  TextEncoder &text_encoder_;
  const std::string model_dir_;
  const bool sdxl_;
  const bool use_v_pred_;

  MNN::Interpreter *safety_interpreter_ = nullptr;
  MNN::Session *safety_session_ = nullptr;
  float nsfw_threshold_ = 0.5f;

 private:
  Conditioning encodePrompts(const GenerationRequest &req);
  xt::xarray<float> encodeImageToLatent(
      const GenerationRequest &req, const xt::xarray<float> &original_image);
  xt::xarray<float> decodeToPixels(const GenerationRequest &req,
                                   const xt::xarray<float> &latents,
                                   bool verbose);
  xt::xarray<float> runUnetTiled(const GenerationRequest &req,
                                 const xt::xarray<float> &latents_scaled,
                                 int timestep, bool skip_uncond,
                                 Conditioning &cond);

  bool useTiledVae(const GenerationRequest &req) const {
    return vaeTilingSupported() &&
           (req.width > vaeTilePixelSize() || req.height > vaeTilePixelSize());
  }
  std::string renderPreview(const GenerationRequest &req,
                            const xt::xarray<float> &latents);

  static std::vector<uint8_t> pixelsToBytes(const xt::xarray<float> &pixels) {
    auto img = xt::view(pixels, 0);
    auto transp = xt::transpose(img, {1, 2, 0});
    auto norm = xt::clip(((transp + 1.0) / 2.0) * 255.0, 0.0, 255.0);
    xt::xarray<uint8_t> u8_img = xt::cast<uint8_t>(norm);
    return std::vector<uint8_t>(u8_img.begin(), u8_img.end());
  }

  // Crops the centered crop_w x crop_h region out of a w x h RGB byte image.
  static void cropCenter(std::vector<uint8_t> &data, int w, int h, int crop_w,
                         int crop_h) {
    int px0 = (w - crop_w) / 2;
    int py0 = (h - crop_h) / 2;
    std::vector<uint8_t> cropped((size_t)3 * crop_w * crop_h);
    for (int y = 0; y < crop_h; ++y) {
      const uint8_t *src_row = data.data() + ((size_t)(py0 + y) * w + px0) * 3;
      uint8_t *dst_row = cropped.data() + (size_t)y * crop_w * 3;
      std::memcpy(dst_row, src_row, (size_t)crop_w * 3);
    }
    data = std::move(cropped);
  }

  static bool needsAspectCrop(const GenerationRequest &req) {
    return req.aspect_pad_inpaint && req.target_crop_width > 0 &&
           req.target_crop_height > 0 &&
           (req.target_crop_width != req.width ||
            req.target_crop_height != req.height);
  }

  static int64_t elapsedMs(
      const std::chrono::high_resolution_clock::time_point &start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now() - start)
        .count();
  }
};

inline Conditioning Pipeline::encodePrompts(const GenerationRequest &req) {
  const int batch_size = 2;
  Conditioning cond;
  cond.hidden_dim =
      sdxl_ ? text_embedding_size + text_embedding_size_2 : text_embedding_size;
  cond.pooled_dim = sdxl_ ? text_embedding_size_2 : 0;
  cond.hidden.assign((size_t)batch_size * 77 * cond.hidden_dim, 0.0f);
  if (sdxl_) {
    cond.pooled.assign((size_t)batch_size * cond.pooled_dim, 0.0f);
    cond.time_ids.assign((size_t)batch_size * 6, 0.0f);
    for (int b = 0; b < batch_size; b++) {
      cond.time_ids[b * 6 + 0] = (float)req.height;  // original_size h
      cond.time_ids[b * 6 + 1] = (float)req.width;   // original_size w
      cond.time_ids[b * 6 + 2] = 0.0f;               // crop_top
      cond.time_ids[b * 6 + 3] = 0.0f;               // crop_left
      cond.time_ids[b * 6 + 4] = (float)req.height;  // target_size h
      cond.time_ids[b * 6 + 5] = (float)req.width;   // target_size w
    }
  }

  // Persistent per-prompt CLIP cache. Positive and negative are looked up
  // independently -- a one-sided hit still saves half the CLIP work. A side
  // whose prompt resolves any TI embedding token is excluded from disk
  // caching: the CLIP output then depends on currently-loaded embedding
  // data we don't want frozen into a stable file.
  std::string cache_dir = ensureCacheDir(model_dir_);
  bool neg_cache_eligible =
      !cache_dir.empty() &&
      !text_encoder_.promptHasEmbedding(req.negative_prompt);
  bool pos_cache_eligible =
      !cache_dir.empty() && !text_encoder_.promptHasEmbedding(req.prompt);

  const uint32_t cache_mode =
      sdxl_ ? prompt_cache::kModeSdxl : prompt_cache::kModeSd15;

  bool neg_hit = neg_cache_eligible &&
                 prompt_cache::load(cache_dir, req.negative_prompt, cache_mode,
                                    cond.hidden_dim, cond.pooled_dim,
                                    cond.negHidden(), cond.negPooled());
  bool pos_hit =
      pos_cache_eligible &&
      prompt_cache::load(cache_dir, req.prompt, cache_mode, cond.hidden_dim,
                         cond.pooled_dim, cond.posHidden(), cond.posPooled());

  if (neg_hit) QNN_INFO("Prompt cache hit (negative)");
  if (pos_hit) QNN_INFO("Prompt cache hit (positive)");

  if (neg_hit && pos_hit) {
    QNN_INFO("CLIP cache hit (both sides), skipping CLIP inference");
    return cond;
  }

  ProcessedPromptPair processed =
      text_encoder_.processPromptPair(req.prompt, req.negative_prompt, 77);

  auto parsed_input_text = text_encoder_.decode(processed.ids);
  QNN_INFO("Parsed Input Text: %s", parsed_input_text.c_str());

  encodeText(processed, !neg_hit, !pos_hit, cond);

  // Persist freshly-computed CLIP outputs (per side). Sides that used a
  // TI embedding stay out of disk cache.
  if (!neg_hit && neg_cache_eligible) {
    prompt_cache::save(cache_dir, req.negative_prompt, cache_mode,
                       cond.hidden_dim, cond.pooled_dim, cond.negHidden(),
                       cond.negPooled());
  }
  if (!pos_hit && pos_cache_eligible) {
    prompt_cache::save(cache_dir, req.prompt, cache_mode, cond.hidden_dim,
                       cond.pooled_dim, cond.posHidden(), cond.posPooled());
  }
  return cond;
}

inline xt::xarray<float> Pipeline::encodeImageToLatent(
    const GenerationRequest &req, const xt::xarray<float> &original_image) {
  std::vector<int> shape = {1, 4, sample_height, sample_width};

  if (!useTiledVae(req)) {
    std::vector<float> vae_enc_mean(1 * 4 * sample_width * sample_height);
    std::vector<float> vae_enc_std(1 * 4 * sample_width * sample_height);

    // For SDXL aspect-ratio padded inpaint with a synthetic base (txt2img
    // path) the VAE encoder input is a deterministic white-on-black canvas
    // keyed by target_crop size, so the (mean, std) latent stats are
    // reproducible. Cache them to disk so we pay the encoder cost only once
    // per (model, target size). User-supplied images (img2img / inpaint) are
    // content-dependent and skip the cache.
    std::string black_latent_cache_path;
    bool loaded_from_cache = false;
    if (req.aspect_pad_inpaint && req.aspect_pad_synthetic_base &&
        !model_dir_.empty()) {
      auto cache_dir = ensureCacheDir(model_dir_);
      if (!cache_dir.empty()) {
        black_latent_cache_path = cache_dir + "/aspect_latent_" +
                                  std::to_string(req.target_crop_width) + "x" +
                                  std::to_string(req.target_crop_height) +
                                  ".bin";
      }
      std::ifstream ifs(black_latent_cache_path, std::ios::binary);
      if (ifs) {
        ifs.seekg(0, std::ios::end);
        std::streamsize sz = ifs.tellg();
        size_t expected =
            (vae_enc_mean.size() + vae_enc_std.size()) * sizeof(float);
        if (sz == (std::streamsize)expected) {
          ifs.seekg(0);
          ifs.read(reinterpret_cast<char *>(vae_enc_mean.data()),
                   vae_enc_mean.size() * sizeof(float));
          ifs.read(reinterpret_cast<char *>(vae_enc_std.data()),
                   vae_enc_std.size() * sizeof(float));
          loaded_from_cache = ifs.good();
          if (loaded_from_cache) {
            std::cout << "Loaded aspect-canvas VAE latent from cache: "
                      << black_latent_cache_path << std::endl;
          }
        }
      }
    }

    if (!loaded_from_cache) {
      vaeEncode(req, req.img_data.data(), vae_enc_mean.data(),
                vae_enc_std.data());

      // Persist the freshly-computed aspect-canvas latent stats for reuse
      // on subsequent runs at the same target size.
      if (req.aspect_pad_inpaint && !black_latent_cache_path.empty()) {
        std::ofstream ofs(black_latent_cache_path, std::ios::binary);
        if (ofs) {
          ofs.write(reinterpret_cast<const char *>(vae_enc_mean.data()),
                    vae_enc_mean.size() * sizeof(float));
          ofs.write(reinterpret_cast<const char *>(vae_enc_std.data()),
                    vae_enc_std.size() * sizeof(float));
          if (ofs.good()) {
            std::cout << "Saved aspect-canvas VAE latent to cache: "
                      << black_latent_cache_path << std::endl;
          }
        }
      }
    }

    auto mean = xt::adapt(vae_enc_mean, shape);
    auto std_dev = xt::adapt(vae_enc_std, shape);
    xt::xarray<float> noise_0 = xt::random::randn<float>(shape);
    return xt::eval(mean + std_dev * noise_0);
  }

  // Tiled path (input larger than the fixed VAE graph).
  std::cout << "Using VAE encoder tiling for " << req.width << "x" << req.height
            << " input..." << std::endl;

  const int vae_enc_tile_size = vaeTilePixelSize();
  const int vae_enc_latent_tile_size = vae_enc_tile_size / 8;

  auto [img_positions, latent_positions, img_overlap_x, img_overlap_y,
        latent_overlap_x, latent_overlap_y] =
      calculate_vae_tile_positions(req.width, req.height, vae_enc_tile_size);

  std::cout << "VAE encoder will use " << img_positions.size()
            << " tiles with overlap " << img_overlap_x << "x" << img_overlap_y
            << "px (latent: " << latent_overlap_x << "x" << latent_overlap_y
            << ")" << std::endl;

  std::vector<std::pair<xt::xarray<float>, xt::xarray<float>>>
      encoded_tiles_mean_std;
  encoded_tiles_mean_std.reserve(img_positions.size());

  {
    // The encoder graph IO is sized from the Config.hpp globals; run it at
    // tile size while this scope is active.
    ScopedSizeOverride tile_io(vae_enc_tile_size, vae_enc_tile_size,
                               vae_enc_latent_tile_size,
                               vae_enc_latent_tile_size);

    for (size_t i = 0; i < img_positions.size(); ++i) {
      auto img_pos = img_positions[i];
      xt::xarray<float> img_tile = xt::view(
          original_image, 0, xt::all(),
          xt::range(img_pos.second, img_pos.second + vae_enc_tile_size),
          xt::range(img_pos.first, img_pos.first + vae_enc_tile_size));

      std::vector<float> tile_img_vec(img_tile.begin(), img_tile.end());
      std::vector<float> tile_mean_vec(1 * 4 * vae_enc_latent_tile_size *
                                       vae_enc_latent_tile_size);
      std::vector<float> tile_std_vec(1 * 4 * vae_enc_latent_tile_size *
                                      vae_enc_latent_tile_size);

      vaeEncode(req, tile_img_vec.data(), tile_mean_vec.data(),
                tile_std_vec.data());

      std::vector<int> tile_shape = {1, 4, vae_enc_latent_tile_size,
                                     vae_enc_latent_tile_size};
      encoded_tiles_mean_std.push_back({xt::adapt(tile_mean_vec, tile_shape),
                                        xt::adapt(tile_std_vec, tile_shape)});
      std::cout << "Processed VAE encoder tile " << i + 1 << "/"
                << img_positions.size() << std::endl;
    }
  }

  xt::xarray<float> img_lat = blend_vae_encoder_tiles(
      encoded_tiles_mean_std, latent_positions, sample_height, sample_width,
      vae_enc_latent_tile_size, latent_overlap_x, latent_overlap_y);

  std::cout << "VAE encoder tiling completed: " << encoded_tiles_mean_std.size()
            << " tiles processed and blended" << std::endl;

  return img_lat;
}

inline xt::xarray<float> Pipeline::decodeToPixels(
    const GenerationRequest &req, const xt::xarray<float> &latents,
    bool verbose) {
  if (!useTiledVae(req)) {
    std::vector<float> vae_dec_in_vec(latents.begin(), latents.end());
    std::vector<float> vae_dec_out_pixels(1 * 3 * req.width * req.height);
    vaeDecode(req, vae_dec_in_vec.data(), vae_dec_out_pixels.data());
    std::vector<int> pixel_shape = {1, 3, req.height, req.width};
    xt::xarray<float> pixels = xt::adapt(vae_dec_out_pixels, pixel_shape);
    return pixels;
  }

  const int vae_tile_size = vaeTilePixelSize();
  const int vae_latent_tile_size = vae_tile_size / 8;

  auto [output_positions, latent_positions, overlap_x, overlap_y,
        latent_overlap_x, latent_overlap_y] =
      calculate_vae_tile_positions(req.width, req.height, vae_tile_size);

  if (verbose) {
    std::cout << "VAE decoder will use " << output_positions.size()
              << " tiles with overlap " << overlap_x << "x" << overlap_y
              << "px (latent: " << latent_overlap_x << "x" << latent_overlap_y
              << ")" << std::endl;
  }

  std::vector<xt::xarray<float>> decoded_tiles;
  decoded_tiles.reserve(latent_positions.size());

  {
    ScopedSizeOverride tile_io(vae_tile_size, vae_tile_size,
                               vae_latent_tile_size, vae_latent_tile_size);

    for (size_t i = 0; i < latent_positions.size(); ++i) {
      auto lat_pos = latent_positions[i];
      xt::xarray<float> latent_tile = xt::view(
          latents, 0, xt::all(),
          xt::range(lat_pos.second, lat_pos.second + vae_latent_tile_size),
          xt::range(lat_pos.first, lat_pos.first + vae_latent_tile_size));

      std::vector<float> tile_latent_vec(latent_tile.begin(),
                                         latent_tile.end());
      xt::xarray<float> tile_output =
          xt::zeros<float>({1, 3, vae_tile_size, vae_tile_size});

      vaeDecode(req, tile_latent_vec.data(), tile_output.data());

      decoded_tiles.push_back(std::move(tile_output));

      if (verbose) {
        std::cout << "Processed VAE tile " << i + 1 << "/"
                  << latent_positions.size() << std::endl;
      }
    }
  }

  xt::xarray<float> pixels =
      blend_output_tiles(decoded_tiles, output_positions, req.height, req.width,
                         vae_tile_size, overlap_x, overlap_y);

  if (verbose) {
    std::cout << "VAE tiling completed: " << decoded_tiles.size()
              << " tiles processed and blended" << std::endl;
  }

  return pixels;
}

// One full tiled UNet evaluation for ultrafix: runs the fixed-size UNet graph
// over an overlapping grid of latent tiles, applies CFG per tile, and blends
// the per-tile noise predictions with the same fade weights as the VAE
// tiling. Blending the predictions (instead of per-tile denoised latents)
// keeps a single scheduler step over the full latent, so stateful schedulers
// (DPM history, ancestral noise) see exactly one image-wide trajectory.
inline xt::xarray<float> Pipeline::runUnetTiled(
    const GenerationRequest &req, const xt::xarray<float> &latents_scaled,
    int timestep, bool skip_uncond, Conditioning &cond) {
  const int tile_px = req.ultrafix_tile;
  const int tile_lat = tile_px / 8;
  const int full_w = sample_width;
  const int full_h = sample_height;
  const int min_overlap = tile_lat / 4;

  auto [positions, overlap_x, overlap_y] =
      calculate_latent_tile_grid(full_w, full_h, tile_lat, min_overlap);

  const int single_size = 4 * tile_lat * tile_lat;
  std::vector<float> tile_in(2 * single_size);
  std::vector<float> tile_out(2 * single_size);
  std::vector<int> tile_shape = {1, 4, tile_lat, tile_lat};

  std::vector<xt::xarray<float>> pred_tiles;
  pred_tiles.reserve(positions.size());

  // The UNet graph IO is sized from the Config.hpp globals; run it at tile
  // size while this scope is active.
  ScopedSizeOverride tile_io(tile_px, tile_px, tile_lat, tile_lat);

  for (auto [x, y] : positions) {
    xt::xarray<float> lat_tile =
        xt::view(latents_scaled, 0, xt::all(), xt::range(y, y + tile_lat),
                 xt::range(x, x + tile_lat));
    std::copy(lat_tile.begin(), lat_tile.end(), tile_in.begin());
    std::copy(lat_tile.begin(), lat_tile.end(), tile_in.begin() + single_size);

    runUnetStep(req, tile_in.data(), timestep, skip_uncond, cond,
                tile_out.data());

    xt::xarray<float> pred;
    if (skip_uncond) {
      std::vector<float> cond_only(tile_out.begin() + single_size,
                                   tile_out.end());
      pred = xt::adapt(cond_only, tile_shape);
    } else {
      std::vector<int> batch2_shape = {2, 4, tile_lat, tile_lat};
      xt::xarray<float> pred_batch = xt::adapt(tile_out, batch2_shape);
      xt::xarray<float> uncond = xt::view(pred_batch, 0);
      xt::xarray<float> txt = xt::view(pred_batch, 1);
      pred = xt::eval(uncond + req.cfg * (txt - uncond));
      pred.reshape({1, 4, tile_lat, tile_lat});
    }
    pred_tiles.push_back(std::move(pred));
  }

  return blend_output_tiles(pred_tiles, positions, full_h, full_w, tile_lat,
                            overlap_x, overlap_y);
}

// Decodes the current latents to a base64 RGB preview, cropped the same way
// as the final image so the UI sees consistent framing. Returns "" when the
// decode fails (the progress event is then sent without an image).
inline std::string Pipeline::renderPreview(const GenerationRequest &req,
                                           const xt::xarray<float> &latents) {
  try {
    xt::xarray<float> preview_latents = xt::eval((1.0 / vaeScale()) * latents);
    xt::xarray<float> pixels =
        decodeToPixels(req, preview_latents, /*verbose=*/false);

    std::vector<uint8_t> out_data = pixelsToBytes(pixels);
    int preview_w = req.width;
    int preview_h = req.height;
    if (needsAspectCrop(req)) {
      cropCenter(out_data, req.width, req.height, req.target_crop_width,
                 req.target_crop_height);
      preview_w = req.target_crop_width;
      preview_h = req.target_crop_height;
    }
    if (req.preview_format == "jpeg") {
      // Quality 75 is plenty for an in-progress preview and keeps the
      // per-step payload small.
      out_data = encodeJPEG(out_data, preview_w, preview_h, 75);
    } else if (req.preview_format == "png") {
      out_data = encodePNG(out_data, preview_w, preview_h);
    }
    std::string image_str_result(out_data.begin(), out_data.end());
    return base64_encode(image_str_result);
  } catch (const std::exception &e) {
    QNN_WARN("Preview generation failed: %s", e.what());
    return "";
  }
}

inline GenerationResult Pipeline::generate(
    GenerationRequest &req, const ProgressCallback &progress_callback) {
  if (req.prompt.empty()) throw std::invalid_argument("Prompt empty");
  if (safety_interpreter_ && !safety_session_)
    throw std::runtime_error("SafetyChecker missing");
  if (req.img2img && !supportsImg2Img())
    throw std::runtime_error("img2img not available (VAE encoder not loaded)");
  if (req.ultrafix && !supportsUltrafix())
    throw std::runtime_error("ultrafix not available on this backend");
  if (req.img2img && req.img_data.size() != (size_t)3 * req.width * req.height)
    throw std::invalid_argument("Invalid img_data");
  if (req.has_mask &&
      (req.mask_data.size() != (size_t)4 * (req.width / 8) * (req.height / 8) ||
       req.mask_data_full.size() != (size_t)3 * req.width * req.height))
    throw std::invalid_argument("Invalid mask_data");

  // The QNN graph helpers size their IO from these globals.
  output_width = req.width;
  output_height = req.height;
  sample_width = req.width / 8;
  sample_height = req.height / 8;

  // Catch-all guard: release any per-request (lowram) model when this
  // function exits, normal return or exception. The explicit release calls
  // inside the hooks stay in place to free memory between pipeline stages.
  ScopeExit transient_guard{[this]() { releaseTransientModels(); }};

  try {
    auto start_time = std::chrono::high_resolution_clock::now();
    int first_step_time_ms = 0;
    int total_run_steps = req.steps + (req.img2img ? 1 : 0) + 2;
    int current_step = 0;
    const int batch_size = 2;

    // --- CLIP ---
    auto clip_start = std::chrono::high_resolution_clock::now();
    Conditioning cond = encodePrompts(req);
    std::cout << "CLIP dur: " << elapsedMs(clip_start) << "ms\n";
    current_step++;
    progress_callback(current_step, total_run_steps, "");

    // --- Scheduler & Latents ---
    const char *timestep_spacing = sdxl_ ? "trailing" : "leading";
    std::unique_ptr<Scheduler> scheduler =
        createScheduler(req.scheduler_type, timestep_spacing);
    if (use_v_pred_) scheduler->set_prediction_type("v_prediction");
    scheduler->set_timesteps(req.steps);
    xt::xarray<float> timesteps = scheduler->get_timesteps();
    const float vae_scale = vaeScale();
    std::vector<int> shape = {1, 4, sample_height, sample_width};
    std::vector<int> shape_batch2 = {batch_size, 4, sample_height,
                                     sample_width};
    xt::random::seed(req.seed);
    xt::xarray<float> latents = xt::random::randn<float>(shape);
    xt::xarray<float> latents_noise = xt::random::randn<float>(shape);

    // Scale initial latents by init_noise_sigma (required for Euler
    // schedulers).
    latents = latents * scheduler->get_init_noise_sigma();

    xt::xarray<float> original_latents, original_image, mask, mask_full;
    int start_step = 0;

    // --- Img2Img / VAE Encode ---
    if (req.img2img) {
      auto vae_enc_start = std::chrono::high_resolution_clock::now();
      std::vector<int> img_shape = {1, 3, req.height, req.width};
      original_image = xt::adapt(req.img_data, img_shape);

      xt::xarray<float> img_lat = encodeImageToLatent(req, original_image);
      xt::xarray<float> img_lat_scaled = xt::eval(vae_scale * img_lat);

      std::cout << "VAE Enc dur: " << elapsedMs(vae_enc_start) << "ms\n";

      original_latents = img_lat_scaled;
      start_step = req.steps * (1.0f - req.denoise_strength);
      // Clamp so timesteps(start_step) below is never out-of-bounds. With
      // denoise_strength = 0 (often used to inspect the base image) the
      // unclamped value would equal `steps` and the OOB read produced
      // garbage noise that decoded to a random pattern.
      if (start_step >= req.steps) start_step = req.steps - 1;
      if (start_step < 0) start_step = 0;
      total_run_steps -= start_step;
      scheduler->set_begin_index(start_step);
      xt::xarray<int> t = {(int)(timesteps(start_step))};

      // For SYNTHETIC-base aspect padding (txt2img path) we replace the
      // mask region with a txt2img-style pure-noise prior so the generated
      // region doesn't inherit the black-canvas bias from VAE encoding.
      // Do NOT do this for user-image base (img2img / inpaint): there the
      // mask region should start from the user's actual image (noised), not
      // pure noise -- otherwise img2img degenerates into txt2img.
      xt::xarray<float> pure_noise_latents;
      if (req.aspect_pad_synthetic_base) {
        pure_noise_latents = xt::eval(latents);
      }

      latents = scheduler->add_noise(original_latents, latents_noise, t);

      if (req.has_mask) {
        mask = xt::adapt(req.mask_data, {1, 4, sample_height, sample_width});
        mask_full =
            xt::adapt(req.mask_data_full, {1, 3, output_height, output_width});

        if (req.aspect_pad_synthetic_base) {
          // Inside the mask: txt2img-style pure noise (no black-latent bias).
          // Outside: noised black latent, kept stable each step by the mask
          // blend further down in the denoising loop.
          latents =
              xt::eval(pure_noise_latents * mask + latents * (1.0f - mask));
        }
      }

      // The full-resolution pixel buffers are only needed again for the
      // post-decode laplacian blend, which runs only with a user mask. Free
      // them otherwise: at ultrafix sizes (e.g. 4096x4096) each one holds
      // ~190 MB through the whole denoising loop.
      if (!req.has_mask) {
        req.img_data = std::vector<float>();
        original_image = xt::xarray<float>();
      }

      current_step++;
      progress_callback(current_step, total_run_steps, "");
    }

    // --- UNET Denoising Loop ---
    int single_latent_size = 1 * 4 * sample_width * sample_height;

    // Ultrafix: the latent is larger than the fixed UNet graph, so every
    // step runs the graph over an overlapping tile grid.
    const int unet_tile_lat = req.ultrafix_tile / 8;
    const bool unet_tiled = req.ultrafix && (sample_width > unet_tile_lat ||
                                             sample_height > unet_tile_lat);
    if (unet_tiled) {
      std::cout << "Ultrafix: tiled UNet at " << req.ultrafix_tile
                << "px tiles over " << req.width << "x" << req.height
                << std::endl;
    }

    beginDenoise(req);

    for (int i = start_step; i < (int)timesteps.size(); ++i) {
      if (req.show_diffusion_process && previewSupported() &&
          (i - start_step) % req.show_diffusion_stride == 0) {
        progress_callback(current_step, total_run_steps,
                          renderPreview(req, latents));
      } else {
        progress_callback(current_step, total_run_steps, "");
      }

      auto step_start_time = std::chrono::high_resolution_clock::now();

      // Scale model input (required for Euler schedulers).
      float current_ts = timesteps(i);
      xt::xarray<float> latents_scaled =
          scheduler->scale_model_input(latents, current_ts);

      const bool skip_uncond = canSkipUncond() && (req.cfg == 1.0f);

      xt::xarray<float> noise_pred;
      if (unet_tiled) {
        noise_pred =
            runUnetTiled(req, latents_scaled, static_cast<int>(current_ts),
                         skip_uncond, cond);
      } else {
        std::vector<float> latents_in_vec;
        latents_in_vec.reserve(batch_size * single_latent_size);
        latents_in_vec.insert(latents_in_vec.end(), latents_scaled.begin(),
                              latents_scaled.end());
        latents_in_vec.insert(latents_in_vec.end(), latents_scaled.begin(),
                              latents_scaled.end());
        std::vector<float> unet_out_latents(batch_size * single_latent_size);

        runUnetStep(req, latents_in_vec.data(), static_cast<int>(current_ts),
                    skip_uncond, cond, unet_out_latents.data());

        if (skip_uncond) {
          // cfg = 1 path: only the cond half of unet_out_latents was filled.
          std::vector<float> cond_only(
              unet_out_latents.begin() + single_latent_size,
              unet_out_latents.end());
          noise_pred = xt::adapt(cond_only, shape);
        } else {
          xt::xarray<float> noise_pred_batch =
              xt::adapt(unet_out_latents, shape_batch2);
          xt::xarray<float> uncond = xt::view(noise_pred_batch, 0);
          xt::xarray<float> txt = xt::view(noise_pred_batch, 1);
          noise_pred = xt::eval(uncond + req.cfg * (txt - uncond));
        }
      }

      auto step_dur = elapsedMs(step_start_time);
      if (i == start_step) first_step_time_ms = (int)step_dur;
      std::cout << "UNET step " << i << " dur: " << step_dur << "ms\n";

      latents = scheduler->step(noise_pred, timesteps(i), latents).prev_sample;

      if (req.has_mask) {
        xt::xarray<int> t_xt = {(int)(timesteps(i))};
        xt::xarray<float> orig_noised =
            scheduler->add_noise(original_latents, latents_noise, t_xt);
        latents = xt::eval(orig_noised * (1.0f - mask) + latents * mask);
      }

      current_step++;
    }

    endDenoise();

    // --- VAE Decode ---
    auto vae_dec_start = std::chrono::high_resolution_clock::now();

    if (useTiledVae(req)) {
      std::cout << "Using VAE decoder tiling for " << req.width << "x"
                << req.height << " output..." << std::endl;
    }

    latents = xt::eval((1.0 / vae_scale) * latents);
    xt::xarray<float> pixels = decodeToPixels(req, latents, /*verbose=*/true);

    std::cout << "VAE Dec dur: " << elapsedMs(vae_dec_start) << "ms\n";

    // --- Post-process Image ---
    // Laplacian-blend the decoded image against the original only when the
    // user actually painted a mask (real inpaint). For auto-installed aspect
    // masks the "original" is just the synthetic canvas / padded user image
    // and the mask region is the entire visible crop, so blending adds no
    // value and risks contaminating with the surrounding canvas.
    if (req.has_mask && req.user_supplied_mask) {
      if (req.aspect_pad_inpaint) {
        // Blend only inside the centered target rectangle so the discarded
        // black border / pad don't pull dark content into the crop edge.
        int px0 = (req.width - req.target_crop_width) / 2;
        int py0 = (req.height - req.target_crop_height) / 2;
        xt::xarray<float> orig_crop =
            xt::eval(xt::view(original_image, 0, xt::all(),
                              xt::range(py0, py0 + req.target_crop_height),
                              xt::range(px0, px0 + req.target_crop_width)));
        xt::xarray<float> gen_crop = xt::eval(xt::view(
            pixels, 0, xt::all(), xt::range(py0, py0 + req.target_crop_height),
            xt::range(px0, px0 + req.target_crop_width)));
        xt::xarray<float> mask_crop =
            xt::eval(xt::view(mask_full, 0, xt::all(),
                              xt::range(py0, py0 + req.target_crop_height),
                              xt::range(px0, px0 + req.target_crop_width)));
        auto blended = laplacianPyramidBlend(orig_crop, gen_crop, mask_crop);
        // Write back into the same target rectangle of `pixels`.
        auto target_view = xt::view(
            pixels, 0, xt::all(), xt::range(py0, py0 + req.target_crop_height),
            xt::range(px0, px0 + req.target_crop_width));
        target_view = xt::reshape_view(
            blended, {3, req.target_crop_height, req.target_crop_width});
      } else {
        auto orig_img_view = xt::view(original_image, 0);  // (3, H, W)
        auto gen_img_view = xt::view(pixels, 0);           // (3, H, W)
        auto mask_view = xt::view(mask_full, 0);           // (1, H, W)

        auto blended =
            laplacianPyramidBlend(orig_img_view, gen_img_view, mask_view);
        pixels = xt::reshape_view(blended, {1, 3, req.height, req.width});
      }
    }

    std::vector<uint8_t> out_data = pixelsToBytes(pixels);

    int final_width = req.width;
    int final_height = req.height;

    // --- Safety Checker ---
    if (safety_interpreter_) {
      auto safety_start = std::chrono::high_resolution_clock::now();
      float score = 0.0f;

      if (safety_check(out_data, req.width, req.height, score,
                       safety_interpreter_, safety_session_)) {
        std::cout << "NSFW Score: " << score << std::endl;
        if (score > nsfw_threshold_) {
          QNN_WARN("NSFW detected (%.2f>%.2f).", score, nsfw_threshold_);
          std::fill(out_data.begin(), out_data.end(), 255);
        }
      } else {
        QNN_WARN("Safety check failed.");
      }

      std::cout << "Safety check dur: " << elapsedMs(safety_start) << "ms\n";
    }

    current_step++;
    progress_callback(current_step, total_run_steps, "");
    auto total_time = elapsedMs(start_time);

    // SDXL aspect-ratio padded inpaint: crop the centered target region out
    // of the 1024x1024 canvas before returning.
    if (needsAspectCrop(req)) {
      cropCenter(out_data, req.width, req.height, req.target_crop_width,
                 req.target_crop_height);
      final_width = req.target_crop_width;
      final_height = req.target_crop_height;
    }

    return GenerationResult{out_data,
                            final_width,
                            final_height,
                            3,
                            static_cast<int>(total_time),
                            first_step_time_ms};
  } catch (const std::exception &e) {
    QNN_ERROR("Image generation error: %s", e.what());
    throw;
  }
}

#endif  // PIPELINE_HPP
