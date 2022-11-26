from models import LungCancerClassifier, LungCancerMalignantClassifier
from utilities import Predict
import cv2

lungCancerClassifier = LungCancerClassifier.LungCancerClassifier()
lungCancerClassification = Predict.model_loader(lungCancerClassifier, 'checkpoint\model_LungCancerClassifier_bs150_lr0.01_epoch9')

lungCancerMalignantClassifier = LungCancerMalignantClassifier.LungCancerMalignantClassifier()
lungCancerMalignantClassification = Predict.model_loader(lungCancerMalignantClassifier, 'checkpoint\model_LungCancerMalignantClassification_bs64_lr0.0065_epoch12')

try:
    inputImage = 'test\lungaca10.jpeg'

    # Image read and display
    inputImg = cv2.imread(inputImage)
    cv2.imshow("Input Lung Histopathological image", inputImg)
    cv2.waitKey()

    # Predict LungCancer Stage
    Predict.classify_image(inputImage, lungCancerClassification, lungCancerMalignantClassification)

except:
    print("Image does not exists")

