module.exports = {
  env: {
    node: true,
    es2021: true
  },
  extends: ['eslint:recommended', 'plugin:prettier/recommended'],
  overrides: [
    {
      env: {
        node: true
      },
      files: ['.eslintrc.{js,cjs}'],
      parserOptions: {
        sourceType: 'script'
      }
    }
  ],
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module'
  },
  rules: {
    indent: ['off', 'tab'],
    quotes: [1, 'single'],
    semi: ['error', 'always'],
    'linebreak-style': [0, 'error', 'windows'],
    'no-unused-vars': 'off',
    'no-empty': 'error',
    'no-console': process.env.NODE_ENV === 'production' ? 'off' : 'off',
    'no-debugger': process.env.NODE_ENV === 'production' ? 'error' : 'off',
    'space-before-function-paren': 'off',
    'vue/multi-word-component-names': 'off',
    'no-class-assign': 'error', //禁止修改类声明的变量
    'no-const-assign': 'error', //禁止修改 const 声明的变量
    'no-dupe-args': 'error', //禁止 function 定义中出现重名参数
    'no-dupe-class-members': 'error', //禁止类成员中出现重复的名称
    'no-func-assign': 'error', //禁止对 function 声明重新赋值
    'no-redeclare': 'error', //禁止多次声明同一变量
    'prettier/prettier': ['warn']
  },
  globals: {
    describe: true,
    it: true,
    before: true,
    beforeEach: true,
    after: true,
    afterEach: true
  }
};
