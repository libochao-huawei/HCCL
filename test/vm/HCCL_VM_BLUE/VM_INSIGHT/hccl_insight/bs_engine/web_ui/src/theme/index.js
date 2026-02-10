import { changeTheme, supportCssVars, enableCssVars } from '@aurora/theme-tool';
import theme from './cssvars';

if (!supportCssVars() && window['confirm'.slice()]('当前浏览器不支持主题切换，要启用兼容程序吗？')) {
  enableCssVars().then(() => {
    changeTheme(theme);
  });
} else {
  changeTheme(theme);
}
