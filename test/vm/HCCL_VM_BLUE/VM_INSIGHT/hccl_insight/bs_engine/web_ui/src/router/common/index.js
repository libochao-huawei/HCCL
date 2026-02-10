import Index from '../../views/index.vue';

export default [
  {
    path: '/',
    component: Index,
    name: 'index',
    redirect: '/checker',
    children: [
      {
        path: '/checker',
        component: () => import('../../views/checker/index.vue'),
        name: 'checker'
      },
      {
        path: '/run-time',
        component: () => import('../../views/runTime/index.vue'),
        name: 'runTime'
      }
    ]
  },
  {
    path: '/login',
    component: () => import('../../views/login/index.vue'),
    name: 'login'
  }
];
