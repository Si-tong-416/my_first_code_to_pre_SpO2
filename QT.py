import subprocess
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit, QLineEdit
from subprocess import Popen, PIPE
class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = '血氧监测'
        self.left = 100
        self.top = 100
        self.width = 400
        self.height = 140
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # 创建按钮
        button = QPushButton('Run SPO2', self)
        button.setToolTip('Click to run SPO2')
        button.move(150, 50)
        button.clicked.connect(self.runSPO2)

        # 创建文本框
        self.textbox = QLineEdit(self)
        self.textbox.move(50, 90)
        self.textbox.resize(300,20)

        self.show()

    def runSPO2(self):
        # 执行SPO2.py文件
        result = subprocess.check_output(['python', 'SPO2.py'])

        # 将结果显示在文本框中
        self.textbox.setText(result.decode("utf-8"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())