    def keyPressEvent(self, event):
        key = event.key()
        print(f"Key pressed: {key} ({event.text()})")

        # 可选：你可以根据按键执行特定操作
        if event.text().lower() == 'q':
            print("Q pressed: quitting application")
            self.close()
