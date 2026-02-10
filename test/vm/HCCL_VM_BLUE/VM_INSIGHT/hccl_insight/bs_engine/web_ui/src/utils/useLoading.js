import { Loading } from '@aurora/vue3';
import { IconLoading } from '@aurora/vue3-icon';

export default function useLoading() {
  let loadingService = null;
  const showLoading = (target) => {
    loadingService = Loading.service({
      spinner: IconLoading(),
      target: target || '.app-container',
      background: 'rgba(0, 0, 0, 0.3)',
      customClass: 'icon-loading'
    });
  };
  const closeLoading = () => loadingService.close();
  return { showLoading, closeLoading };
}
