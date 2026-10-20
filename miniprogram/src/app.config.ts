export default defineAppConfig({
  pages: [
    'pages/index/index',
    'pages/tech/index',
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
        text: '识别',
        iconPath: 'assets/tabbar/home.svg',
        selectedIconPath: 'assets/tabbar/home-selected.svg'
      },
      {
        pagePath: 'pages/tech/index',
        text: '病害知识',
        iconPath: 'assets/tabbar/tech.svg',
        selectedIconPath: 'assets/tabbar/tech-selected.svg'
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
    navigationBarTitleText: '玉米医生 · 病害识别',
    navigationBarTextStyle: 'white'
  }
})
