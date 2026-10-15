/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Dark clinical palette: deep slate surfaces with a cyan/teal accent.
        ink: {
          950: "#05070c",
          900: "#0a0e15",
          850: "#0e131c",
          800: "#131a26",
          700: "#1b2433",
          600: "#273244",
          500: "#3a485f",
        },
        accent: {
          DEFAULT: "#38d1c4",
          soft: "#7fe3da",
          dim: "#1f7f78",
        },
        severity: {
          error: "#ff6b6b",
          warning: "#f0b429",
          info: "#57b8ff",
          ok: "#4ade80",
        },
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
