import { ShaderMount } from "./vendor/paper-shaders/shader-mount.js";
import { waterFragmentShader } from "./vendor/paper-shaders/shaders/water.js";
import { getShaderColorFromString } from "./vendor/paper-shaders/get-shader-color-from-string.js";
import { ShaderFitOptions, defaultObjectSizing } from "./vendor/paper-shaders/shader-sizing.js";
import { emptyPixel } from "./vendor/paper-shaders/empty-pixel.js";

const PARAMS = {
  image: "https://paper.design/flowers.webp",
  colorBack: "#8f8f8f",
  colorHighlight: "#ffffff",
  highlights: 0.4,
  layering: 0,
  edges: 0,
  waves: 0,
  caustic: 0.2,
  size: 0.7,
  speed: 0.1,
  fit: "cover",
  scale: 1,
};

function loadImage(src, crossOrigin) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    if (crossOrigin) {
      image.crossOrigin = "anonymous";
    }
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = src;
  });
}

async function mountWaterBackground() {
  const container = document.createElement("div");
  container.className = "page-water";
  container.setAttribute("aria-hidden", "true");

  let image;
  try {
    image = await loadImage(PARAMS.image, true);
  } catch {
    image = await loadImage(emptyPixel);
  }

  document.body.prepend(container);

  const sizing = { ...defaultObjectSizing, fit: PARAMS.fit, scale: PARAMS.scale };
  const uniforms = {
    u_image: image,
    u_colorBack: getShaderColorFromString(PARAMS.colorBack),
    u_colorHighlight: getShaderColorFromString(PARAMS.colorHighlight),
    u_highlights: PARAMS.highlights,
    u_layering: PARAMS.layering,
    u_edges: PARAMS.edges,
    u_waves: PARAMS.waves,
    u_caustic: PARAMS.caustic,
    u_size: PARAMS.size,
    u_fit: ShaderFitOptions[sizing.fit],
    u_scale: sizing.scale,
    u_rotation: sizing.rotation,
    u_originX: sizing.originX,
    u_originY: sizing.originY,
    u_offsetX: sizing.offsetX,
    u_offsetY: sizing.offsetY,
    u_worldWidth: sizing.worldWidth,
    u_worldHeight: sizing.worldHeight,
  };

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  try {
    new ShaderMount(
      container,
      waterFragmentShader,
      uniforms,
      undefined,
      reducedMotion ? 0 : PARAMS.speed,
      0,
      1.5,
      1920 * 1080,
      ["u_image"]
    );
  } catch (error) {
    container.remove();
    console.warn("Paper water background unavailable:", error);
  }
}

mountWaterBackground();