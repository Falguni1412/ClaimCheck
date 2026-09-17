/** @type {import('next').NextConfig} */
export default {
  reactStrictMode: false,
  swcMinify: true,
  images: {
    domains: [],
  },
  async headers() {
    return [
      {
        source: "/:all",
        headers: [
          {
            key: "X-Content-Type-Options",
            value: "nosniff",
          },
          {
            key: "X-Frame-Options",
            value: "DENY",
          },
          {
            key: "Referrer-Policy",
            value: "origin-when-cross-origin",
          },
        ],
      },
    ]
  },
}