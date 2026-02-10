import { defineComponent, onUnmounted } from 'vue';
import useLoading from '@/utils/useLoading';

export default defineComponent({
  setup() {
    const { showLoading, closeLoading } = useLoading();
    showLoading();
    onUnmounted(() => closeLoading());
    return () => null;
  }
});
