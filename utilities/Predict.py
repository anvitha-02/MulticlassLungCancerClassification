import PIL.Image
import matplotlib.pyplot as plt
import models.LungCancerClassifier
import numpy as np
import torch
import utilities.Utility
from torch.autograd import Variable
from torchvision import models, transforms


def all_models(data, lungCancerClassification, lungCancerMalignantClassification, use_cuda=False):
    for images, labels in data:
        for element, lab in zip(images, labels):
            res = ""
            element = element.unsqueeze(0)
            if use_cuda:
                out2 = torch.sigmoid(lungCancerClassification(element.cuda()))
            else:
                out2 = torch.sigmoid(lungCancerClassification(element))
            if out2[0] < 0.5:
                res = "Lung: Benign"
            else:
                if use_cuda:
                    out4 = torch.sigmoid(lungCancerMalignantClassification(element.cuda()))
                else:
                    out4 = torch.sigmoid(lungCancerMalignantClassification(element))
                if out4[0] < 0.5:
                    res = "Lung: Malignant - ACA"
                else:
                    res = "Lung: Malignant - SCC"
            print(res)


def classify_image(img_path, lungCancerClassification, lungCancerMalignantClassification, use_cuda=False):
    element = image_loader(img_path)
    res = ""
    if use_cuda:
        out2 = torch.sigmoid(lungCancerClassification(element.cuda()))
    else:
        out2 = torch.sigmoid(lungCancerClassification(element))
        # print("LCC: ", out2)
    if out2[0] < 0.5:
        res = "Lung: Benign"
    else:
        if use_cuda:
            out4 = torch.sigmoid(lungCancerMalignantClassification(element.cuda()))
        else:
            out4 = torch.sigmoid(lungCancerMalignantClassification(element))
            # print("LCMC: ", out4)
        if out4[0] < 0.5:
            res = "Lung: Malignant - ACA"
        else:
            res = "Lung: Malignant - SCC"

    print(res)


def image_loader(image_name, use_cuda=False):
    """
    Load image and prepares it for model.

    Args:
        image_name: string representing path of image location
    Returns:
        image.cuda: CUDA tensor
    """
    image = PIL.Image.open(image_name)
    data_transform2 = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = data_transform2(image).float()
    image = image.unsqueeze(0)
    image = Variable(image)
    if use_cuda:
        return image.cuda()
    else:
        return image


def predict(net, model_path, sample_image_path, use_cuda=False):
    """
    Predicts output given a trained neural net model and an image.

    Args:
        net -> PyTorch neural network object
        model_path -> return value from get_model_name()
        sample_image_path -> return value from image_loader()

    Returns:
        tensor: CUDA tensor object

    >>> net = models.LungCancerClassifier.LungCancerClassifier()
    >>> model_path = utilities.Utility.get_model_name(net.name, batch_size=150, learning_rate=0.01, epoch=9)
    >>> img_path = '/content/CNN2_LungClassifierData/test/benign/lungn1019.jpeg'
    >>> predict(net, model_path, img_path)

    Console:
        tensor([0.0005], device='cuda:0', grad_fn=<SigmoidBackward>) -> Prediction: Class 0 (Benign)
    """
    state = torch.load(model_path)
    net.load_state_dict(state)
    if use_cuda:
        net = net.cuda()
    img = image_loader(sample_image_path)
    x = net(img)
    return torch.sigmoid(x)


def model_loader(net, model_path, use_cuda=False):
    if not use_cuda:
        state = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        state = torch.load(model_path)
    net.load_state_dict(state)
    if use_cuda:
        net = net.cuda()
    return net


def connect_accuracy(data, lungCancerClassification, lungCancerMalignantClassification, print_img=False, use_cuda=False):
    total = 0
    corr = 0
    false_n_p = np.array([["pred |label", "la", "ln", "ls"],
                          ["lung_aca ", 0, 0, 0, 0, 0],
                          ["lung_n.  ", 0, 0, 0, 0, 0],
                          ["lung_scc ", 0, 0, 0, 0, 0]
                          ])

    for images, labels in data:
        for element, lab in zip(images, labels):
            lab = int(lab)
            label_string = ''
            pred_string = ''
            if print_img == True:
                p = element.numpy()
                print_plot = plt.figure()
                p = np.transpose(p, (1, 2, 0))
                plt.imshow(p)
            classes = ['lung_aca', 'lung_n', 'lung_scc']
            element = element.unsqueeze(0)

            if use_cuda:
                out2 = torch.sigmoid(lungCancerClassification(element.cuda()))
            else:
                out2 = torch.sigmoid(lungCancerClassification(element))

            if out2[0] < 0.5:
                res = 3
            else:
                if use_cuda:
                    out4 = torch.sigmoid(lungCancerMalignantClassification(element.cuda()))
                else:
                    out4 = torch.sigmoid(lungCancerMalignantClassification(element))
                if out4[0] < 0.5:
                    res = 2
                else:
                    res = 4

            label_string = classes[lab]
            pred_string = classes[res]
            if print_img == True:
                plt.title('Label: ' + label_string + " | Pred: " + pred_string)
            false_n_p[res + 1, lab + 1] = int(false_n_p[res + 1, lab + 1]) + 1
            total += 1
            if res == lab:
                corr += 1
    print(false_n_p)
    print("correct:", corr, ", total:", total, ", accuracy:", corr / total)
