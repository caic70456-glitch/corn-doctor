export default defineAppConfig({
  pages: [
    'pages/index/index',
    'pages/tech/index',
    'pages/deploy/index',
    'pages/about/index'
  ],
  tabBar: {
    color: '#86909c',
    selectedColor: '#2f9e44',
    backgroundColor: '#ffffff',
    borderStyle: 'white',
    list: [
      {
        pagePath: 'pages/index/index',
        text: '概览',
        iconPath: 'assets/tabbar/home.svg',
        selectedIconPath: 'assets/tabbar/home-selected.svg'
      },
      {
        pagePath: 'pages/tech/index',
        text: '技术',
        iconPath: 'assets/tabbar/tech.svg',
        selectedIconPath: 'assets/tabbar/tech-selected.svg'
      },
      {
        pagePath: 'pages/deploy/index',
        text: '部署',
        iconPath: 'assets/tabbar/deploy.svg',
        selectedIconPath: 'assets/tabbar/deploy-selected.svg'
      },
      {
        pagePath: 'pages/about/index',
        text: '关于',
        iconPath: 'assets/tabbar/about.svg',
        selectedIconPath: 'assets/tabbar/about-selected.svg'
      }
    ]
  },
  window: {
    backgroundTextStyle: 'light',
    navigationBarBackgroundColor: '#1e5c33',
    navigationBarTitleText: '玉米医生 · 三维定位方案',
    navigationBarTextStyle: 'white'
  }
})
