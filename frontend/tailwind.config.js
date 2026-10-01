/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#5F9FB5',
        navy: '#173B4D',
        teal: '#4F8F98',
        mist: '#DCECF2',
        canvas: '#F3F8FA',
        muted: '#64808B',
        low: '#4CAF7D',
        moderate: '#E5A84B',
        high: '#D96B6B',
      },
      fontFamily: {
        sans: ['DM Sans', 'sans-serif'],
        display: ['DM Serif Display', 'serif'],
      },
      boxShadow: {
        quiet: '0 8px 26px rgba(23, 59, 77, 0.055)',
      },
    },
  },
  plugins: [],
}