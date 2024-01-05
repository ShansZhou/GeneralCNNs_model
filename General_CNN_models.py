import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



class LeNet():
    def __init__(self, out_feats):
        self.X =[]
        self.Y =[]
        self.cost = 0.0
        self.batchSize = 4
        
        # CNN Kernerl
        self.W_6x5x5= np.random.randn(6,5,5)
        self.W_16x5x5= np.random.randn(16,5,5)
        
        # FC kernerl
        self.W_nx120= np.random.randn(5*5*16, 120)
        self.W_120x84= np.random.randn(120, 84)
        self.W_84xn= np.random.randn(84, out_feats)

    
    def CNN_features_layer(self, x):

        # CNN layer1: Kernel=[B,5,5]  stride=1
        feats1 = self.conv(data=x, kernel=self.W_6x5x5, stride=1, padding=0)
        # Average pooling: f=2, stride=2
        feats1_avg = self.avgPool(data=feats1, kernelSize=2, stride=2)
        
        # CNN layer2: Kernel=[B,5,5]  stride=1
        feats2 = self.conv(data=feats1_avg, kernel=self.W_16x5x5, stride=1, padding=0)
        feats2 = feats2[:,0:16,:,:]
        # Average pooling: f=2, stride=2
        feats2_avg = self.avgPool(data=feats2, kernelSize=2, stride=2)
        
        return feats2_avg
            
    # in_features:[1, n]
    def FC_classifier_layer(self, in_features):
        
        # FC layer: 120
        fc1 = np.dot(in_features, self.W_nx120)
        fc1_relu = self.reLU(fc1)
        
        # FC layer: 84
        fc2 = np.dot(fc1_relu, self.W_120x84) 
        fc2_relu = self.reLU(fc2)
        
        # Output
        output = np.dot(fc2_relu, self.W_84xn)
        
        return self.softmax(output)
        
    def reLU(self, x):
        return np.where(x>0,x,0)
            
    def softmax(self, x):
        e_x = np.exp(x-np.expand_dims(np.max(x, axis=1),1))
        return e_x / np.sum(e_x, axis=0)
    
    # padding part is 0
    def paddingData(self, data, padding):
        
        batch, chs, h, w = np.shape(data)
        
        h_pad = h + 2*padding
        w_pad = w + 2*padding
        
        data_pad = np.zeros((batch, chs, h_pad, w_pad))
        
        for b in range(batch):
            for c in range(chs):
                
                for row in range(h_pad):
                    for col in range(w_pad):
                        
                        # padding part is all equal to 0
                        if row < padding or row > h or col < padding or col > w : continue
                        
                        # otherwise, keep same
                        data_pad[b, c, row, col] = data[b, c, row, col]
        
        return data_pad

    # average pool
    def avgPool(self, data, kernelSize, stride):
        data_batch, data_chs, data_h, data_w = np.shape(data)
        k_h, k_w = kernelSize, kernelSize
        out_h = np.uint16((data_h - k_h) / stride) + 1
        out_w = np.uint16((data_w - k_w) / stride) + 1
        
        avgPool_mat = np.zeros((data_batch, data_chs, out_h, out_w), np.float32)
        offX = k_w//2
        offY = k_h//2
        
        for b in range(data_batch):
            for chs in range(data_chs):
                
                for row in range(out_h):
                    for col in range(out_w):
                        acc = 0.0
                        # accumlate all the element in data within Kernerl
                        for x in range(k_h):
                            for y in range(k_w):
                                data_col = col+x-offX
                                data_row = row+y-offY
                                # skip index out of bounds
                                if data_col < 0 or data_col >= out_w or data_row <0 or data_row > out_h: continue
                                acc+= data[b, chs, data_row, data_col]

                        avgPool_mat[b,chs,row,col] = acc / (k_h*k_w)
            
        return avgPool_mat
        
    def conv(self, data, kernel, stride, padding):
        # padding data
        data_pad = self.paddingData(data, padding)
        
        #
        data_batch, data_chs, data_h, data_w = np.shape(data)
        
        k_chs, k_h, k_w = np.shape(kernel)
        out_h = np.uint16((data_h - k_h + 2*padding) / stride) + 1
        out_w = np.uint16((data_w - k_w + 2*padding) / stride) + 1
        
        conv_mat = np.zeros((data_batch, data_chs*k_chs, out_h, out_w))
        offX = k_w//2
        offY = k_h//2
        
        for b in range(data_batch):
            for c in range(data_chs*k_chs):
                
                for row in range(out_h):
                    for col in range(out_w):
                        
                        acc = 0.0
                        # accumlate all the element-wise multiple
                        for x in range(k_w):
                            for y in range(k_h):
                                data_col = col+x-offX
                                data_row = row+y-offY
                                # skip index out of bounds
                                if data_col < 0 or data_col >= out_w or data_row <0 or data_row > out_h: continue
                                
                                acc+= (data_pad[b, c//k_chs,data_row, data_col]*kernel[c//data_chs,x,y])

                        conv_mat[b, c, row, col] = acc
        
        return conv_mat
                                     
    def train(self, x_train, y_train):

        # forward
        features = self.CNN_features_layer(x_train)
        # flatten features into series
        features_flatten = np.reshape(features, (self.batchSize, 5*5*16))
        y_predict = self.FC_classifier_layer(features_flatten)
        
              
    def predict(self, x):
        features = self.CNN_features_layer(x)
        # flatten features into series
        features_flatten = np.reshape(features, (self.batchSize, 5*5*16))
        y_predict = self.FC_classifier_layer(features_flatten)
        y_labels = np.argmax(y_predict, axis=1)
        
        return y_labels
        