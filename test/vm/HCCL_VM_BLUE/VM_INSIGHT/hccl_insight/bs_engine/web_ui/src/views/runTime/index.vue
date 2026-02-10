<template>
  <div class="content">runtime</div>
</template>

<script setup>
import { onMounted, onUpdated, ref, computed } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { IconPlusSquare, IconEditorList } from '@aurora/vue3-icon';
import { TreeMenu as AuiTreeMenu, Button, Breadcrumb, BreadcrumbItem, Tabs, TabItem, Container, Layout, NavMenu } from '@aurora/vue3';

const route = useRoute();
const router = useRouter();
const AuiButton = Button;
const AuiBreadcrumb = Breadcrumb;
const AuiBreadcrumbItem = BreadcrumbItem;
const AuiTabs = Tabs;
const AuiTabItem = TabItem;
const AuiContainer = Container;
const AuiLayout = Layout;
const AuiNavMenu = NavMenu;

const menuData = [
  {
    title: 'Checker',
    url: '/checker'
  },
  {
    title: 'RunTime',
    url: '/runtime'
  }
];
</script>

<style scoped lang="scss">
.content {
  height: 100%;
  overflow: hidden;
}
</style>
