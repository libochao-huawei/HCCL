import { defineConfig } from 'vite';
import path, { resolve } from 'path';
import vue from '@vitejs/plugin-vue';
import vueJsx from '@vitejs/plugin-vue-jsx';
import eslintPlugin from 'vite-plugin-eslint';
import { createSvgIconsPlugin } from 'vite-plugin-svg-icons';

// https://vitejs.dev/config/
export default defineConfig({
  // transpileDependencies: true,
  // linkOnSave: false,
  // 打包路径 相对路径
  base: './',
  plugins: [
    vue(),
    vueJsx(),
    createSvgIconsPlugin({
      iconDirs: [resolve(process.cwd(), 'src/assets/icons')],
      symbolId: 'icon-[dir]-[name]'
    }),
    eslintPlugin({
      include: ['src/**/*.js', 'src/**/*.vue', 'src/*.js', 'src/*.vue']
    })
  ],
  define: {
    'process.env': Object.assign({}, process.env)
    // global: 'window' // 最新 版本 AUI5 修复了该问题，发包后可以去掉该规则
  },
  worker: {
    format: 'es'
  },
  server: {
    host: 'localhost.huawei.com',
    // host: '0.0.0.0',
    port: 8080,
    // 代理配置  需要将接口ip映射到target的域名
    proxy: {
      '/hccl': {
        // target: 'http://10.189.192.27:4000',
        // target: 'http://10.185.8.155:4000',
        target: 'http://7.183.245.239:4000',
        // target: 'http://7.183.245.239:20008',
        // target: 'http://7.240.216.218:20007',
        changeOrigin: true
      }
    }
  },
  resolve: {
    alias: {
      '@aurora/theme': '@aurora/theme-tool/aui3',
      '@aurora/vue3-icon': '@aurora/vue3-icon-saas',
      '@': path.resolve(__dirname, './src')
    },
    extensions: ['.js', '.ts', '.jsx', '.tsx', '.json', '.vue']
  },
  build: {
    // 规定触发警告的 chunk 大小
    chunkSizeWarningLimit: 10000,
    // target: 'modules', // 设置最终构建的浏览器兼容目标。modules:支持原生 ES 模块的浏览器
    // outDir: 'dist', // 指定输出路径
    // assetsDir: 'assets', // 指定生成静态资源的存放路径
    minify: 'terser', // 混淆器，启用 terser 压缩, terser构建后文件体积更小
    terserOptions: {
      compress: {
        pure_funcs: ['console.log'], // 只删除 console.log
        // drop_console: true, // 删除所有 console
        drop_debugger: true // 删除 debugger
      }
    },
    rollupOptions: {
      output: {
        dir: 'dist',
        // 静态资源打包做处理
        chunkFileNames: 'static/js/[name].[hash].js',
        entryFileNames: 'static/js/[name].[hash].js',
        assetFileNames: 'static/[ext]/[name].[hash].[ext]'
      }
    }
  }
  // 打包单个组件配置
  // build:
  // {
  //   outDir: 'lib',
  //   lib: {
  //     entry: resolve(__dirname, 'packages/index.js'), // 指定组件编译入口文件
  //     name: 'MyUi',
  //     fileName: 'my-ui',
  //   },// 库编译模式配置
  //   rollupOptions: {
  //     external: ['vue'],
  //     output: {
  //       globals: {
  //         vue: 'Vue',
  //       },
  //     },
  //   },
  // }
});
