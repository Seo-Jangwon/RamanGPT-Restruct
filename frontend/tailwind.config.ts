import type { Config } from 'tailwindcss'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'raman': {
          50: '#e6f7ff',
          100: '#b3e5ff',
          200: '#80d4ff',
          300: '#4dc2ff',
          400: '#1ab1ff',
          500: '#0096e6',
          600: '#0075b3',
          700: '#005580',
          800: '#00344d',
          900: '#00141a',
        },
        'sers': {
          50: '#fff5e6',
          100: '#ffe0b3',
          200: '#ffcc80',
          300: '#ffb74d',
          400: '#ffa31a',
          500: '#ff8f00',
          600: '#e67e00',
          700: '#cc6e00',
          800: '#b35d00',
          900: '#994d00',
        },
        'afm-primary': {
          50: '#fef2f2',
          100: '#fee2e2',
          200: '#fecaca',
          300: '#fca5a5',
          400: '#f87171',
          500: '#71141E',
          600: '#5a1018',
          700: '#430c12',
          800: '#2c080c',
          900: '#150406',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
} satisfies Config

