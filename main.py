
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models



def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image is <= 400 pixels in the x-y dims.'''
    
    image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image



def im_convert(tensor):
    ''' helper function for un-normalizing an image and converting it from a Tensor image to a NumPy image for display '''
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def imshow(img):              
    ''' Pour afficher une image '''
    plt.figure(1)
    plt.imshow(img)
    plt.show()






################################################### VGG FEATURES #####################################################
def get_features(image, model, layers=None):
    ''' 
        Run an image forward through a model and get the features for a set of layers
    '''
    
    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv0',
                  '1': 'conv1',
                  '2': 'conv2',
                  '3': 'conv3', 
                  '4': 'conv4', 
                  '5': 'conv5', 
                  '19': 'conv19',   ## content representation
                }

    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x            
    return features



def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    _, d, w, h = tensor.size()
    resized_gram = tensor.view(d, w*h)
    transpose = resized_gram.t()
    gram = torch.mm(resized_gram, transpose)
    return gram


def calc_style_loss(target, style_grams):
    style_l = 0

    for layer in style_grams:
        if style_grams[layer] is not 'conv19':
            #Calculating the gram matrix for the style and the generated image
            G = gram_matrix(target[layer])
            A = style_grams[layer]
                
            #Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
            style_l += torch.mean((G-A)**2) 

    return style_l


if __name__ == '__main__':

    ##########################" VGG "#########################################################""
    # get the "features" portion of VGG19 (we will not need the "classifier" portion)
    vgg = models.vgg19(pretrained=True).features

    # freeze all VGG parameters since we're only optimizing the target image
    for param in vgg.parameters():
        param.requires_grad_(False)


    # move the model to GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    vgg.to(device)

    features = list(vgg)[:23]
    for i,layer in enumerate(features):
        print(i,"   ",layer)


    ########################## DISPLAY IMAGE#########################################################""
    content = load_image('images/montagne_small.jpg').to(device)
    style = load_image('images/peinture1_small.jpg', shape=content.shape[-2:]).to(device)

    #imshow(im_convert(content))
    #imshow(im_convert(style))

    _, d, h, w = content.size()
    print("content size=",d,h,w)
    _, d, h, w = style.size()
    print("style size=",d,h,w)


    # get content and style features only once before training
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    print(type(content_features))
    for key, value in content_features.items():
        print("key=",key)
        #print("value shape=", type(value))
        vnp = value.to("cpu").clone().detach().numpy()
        print("value shape=", vnp.shape)

    target = content.clone().requires_grad_(True).to(device)

    show_every = 100
    optimizer = optim.Adam([target], lr=0.003)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}           # dict with each gram matrix for each feature name
    style_layers = {  'conv0', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5' }


    for i in range(2000):
        content_loss = style_loss = 0
        # TODO get the features from your target image
        target_features = get_features(target, vgg)
        # TODO the content loss
        content_loss += torch.mean((target_features['conv19'] - content_features['conv19'])**2)
        # TODO the style loss
        style_loss += calc_style_loss(target_features, style_grams)
        # TODO calculate the *total* loss
        total_loss =  0.5*content_loss + 0.5*style_loss
        # TODO update your target image

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print('Total loss: ', i, total_loss.item())
        if  i % show_every == 0:
            #imshow(im_convert(target))
            plt.imsave('images/output.png', im_convert(target))
            print("save %d" % i)

