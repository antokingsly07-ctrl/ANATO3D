import "@testing-library/jest-dom/vitest";

// jsdom ships no canvas backend. Provide just enough of the 2D API (and
// ImageData) that the slice viewer can be exercised without a rasteriser.
if (typeof globalThis.ImageData === "undefined") {
  class ImageDataPolyfill {
    data: Uint8ClampedArray;
    width: number;
    height: number;

    constructor(a: number | Uint8ClampedArray, b: number, c?: number) {
      if (typeof a === "number") {
        this.width = a;
        this.height = b;
        this.data = new Uint8ClampedArray(a * b * 4);
      } else {
        this.data = a;
        this.width = b;
        this.height = c ?? Math.floor(a.length / (b * 4));
      }
    }
  }
  // @ts-expect-error assigning a minimal polyfill onto the global scope
  globalThis.ImageData = ImageDataPolyfill;
}

const stubContext = {
  putImageData: () => undefined,
  clearRect: () => undefined,
  drawImage: () => undefined,
  fillRect: () => undefined,
};

// @ts-expect-error jsdom's getContext returns null; substitute a stub
HTMLCanvasElement.prototype.getContext = () => stubContext;
