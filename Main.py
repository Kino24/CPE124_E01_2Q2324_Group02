import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
import requests
from io import BytesIO
from PyQt5.QtCore import Qt

class SkinDiseasePredictorApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Skin Disease Predictor')
        self.setGeometry(100, 100, 600, 400)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 300)

        self.upload_button = QPushButton('Upload Image', self)
        self.upload_button.setFixedSize(150, 30)
        self.upload_button.clicked.connect(self.upload_image)

        self.predict_button = QPushButton('Predict', self)
        self.predict_button.setFixedSize(150, 30)
        self.predict_button.clicked.connect(self.predict_image)
        
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True) 

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)
        
        self.setLayout(layout)

    def upload_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap)

            # Save the file name for later use
            self.file_name = file_name

    def predict_image(self):
        if hasattr(self, 'file_name'):
            try:
                files = {'file': open(self.file_name, 'rb')}
                response = requests.post('http://127.0.0.1:8000/predict', files=files)
                result = response.json()

                predicted_class = result['class']
                confidence = result['confidence']
                result_text = f'Predicted Class: {predicted_class}\nConfidence: {confidence:.2%}'
                self.result_label.setText(result_text)
            except Exception as e:
                self.result_label.setText(f'Error during prediction:\n{str(e)}')
        else:
            self.result_label.setText('Please upload an image first.')


#u can try using this thing below kung gusto mo ng pop up window chuchu ng result:

                #QMessageBox.information(self, 'Prediction Result',
                                        #f'Predicted Class: {predicted_class}\nConfidence: {confidence:.2%}')
            #except Exception as e:
                #QMessageBox.critical(self, 'Error', f'Error during prediction:\n{str(e)}')
        #else:
            #QMessageBox.warning(self, 'Warning', 'Please upload an image first.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = SkinDiseasePredictorApp()
    main_app.show()
    sys.exit(app.exec_())
    #infoBox.exec()
