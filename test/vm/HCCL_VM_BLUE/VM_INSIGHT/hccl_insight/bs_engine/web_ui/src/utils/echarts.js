import * as echarts from 'echarts/core';
import { TooltipComponent, GridComponent, LegendComponent, DataZoomComponent } from 'echarts/components';
import { BarChart, CustomChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([TooltipComponent, GridComponent, LegendComponent, DataZoomComponent, BarChart, CanvasRenderer, CustomChart]);

export default echarts;
