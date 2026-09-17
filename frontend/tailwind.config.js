/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
    },
    extend: {
      colors: {
        claim: {
          100: "#f0f4ff",
          200: "#e4e8f8",
          300: "#d1d8f0",
          400: "#a3b1e8",
          500: "#7b2cbf",
          600: "#5a24a0",
          700: "#4a1d8c",
          800: "#3a166d",
          900: "#2a0f55",
        },
      },
    },
  },
  plugins: [],
};