import { describe, it, expect } from "vitest";
import { pairComponents, type LocalModel } from "../AdminPanel";

const f = (name: string, component: boolean): LocalModel =>
  ({ name, path: `/m/${name}`, size_mb: 1, component });

// Both Qwen sets present at once — the situation that produced a real failure: selecting the 2.1
// model kept v1's VAE and encoder, and every generation died with "model metadata validation failed".
const disk: LocalModel[] = [
  f("qwen_image_2.1-Q4_K.gguf", false),
  f("qwen_image_2.1_vae_bf16.safetensors", true),
  f("Qwen3VL-8B-Instruct-Q4_K_M.gguf", true),
  f("Qwen_Image-Q4_K_M.gguf", false),
  f("Qwen_Image-VAE.safetensors", true),
  f("Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf", true),
  f("sdxl_turbo.safetensors", false),
];

describe("pairComponents", () => {
  it("pairs a 2.1 model with the 2.1 VAE and Qwen3-VL encoder", () => {
    const { vae, textEncoder } = pairComponents("qwen_image_2.1-Q4_K.gguf", disk);
    expect(vae).toBe("/m/qwen_image_2.1_vae_bf16.safetensors");
    expect(textEncoder).toBe("/m/Qwen3VL-8B-Instruct-Q4_K_M.gguf");
  });

  it("pairs the v1 model with v1 parts, not the 2.1 ones", () => {
    const { vae, textEncoder } = pairComponents("Qwen_Image-Q4_K_M.gguf", disk);
    expect(vae).toBe("/m/Qwen_Image-VAE.safetensors");
    expect(textEncoder).toBe("/m/Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf");
  });

  it("returns empty strings when no matching parts exist, so stale paths get cleared", () => {
    const { vae, textEncoder } = pairComponents("qwen_image_9.9-Q4_K.gguf", disk);
    expect(vae).toBe("");
    expect(textEncoder).toBe("");
  });

  it("never offers another model file as a component", () => {
    const { vae, textEncoder } = pairComponents("sdxl_turbo.safetensors", [f("sdxl_turbo.safetensors", false)]);
    expect(vae).toBe("");
    expect(textEncoder).toBe("");
  });
});
