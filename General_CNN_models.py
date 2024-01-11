import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



class LeNet():
    def __init__(self, out_feats):
        self.X =[]
        self.Y =[]
        self.cost = 0.0
        self.batchSize = 4
        self.learn_rate = 0.001
        self.classes = 10
        
        # feat map conbination 6x16
        self.FTtable = np.array([[1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1],
                                [1,1,0,0,0,1,1,1,1,0,1,1,1,1,0,1],
                                [1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1],
                                [0,1,1,1,0,0,1,1,1,1,1,0,1,0,1,1],
                                [0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1],
                                [0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1]])
        
        # CNN Kernerl
        self.W_6x5x5= self.xavierInit(c_in=1,c_out=6,h=5,w=5)
        self.W_16x5x5= self.xavierInit(c_in=6,c_out=16,h=5,w=5)
        
        # FC kernerl
        self.W_nx120= self.xavierInit(c_in=5*5*16,c_out=120,h=1,w=1, fc=True)
        self.W_120x84= self.xavierInit(c_in=120,c_out=84,h=1,w=1, fc=True) 
        self.W_84xn= self.xavierInit(c_in=84,c_out=out_feats,h=1,w=1, fc=True)
    
    def CNN_features_layer(self, x):

        # CNN layer1: Kernel=[B,5,5]  stride=1
        feats1 = self.conv(data=x, kernel=self.W_6x5x5, stride=1, padding=0)
        # Average pooling: f=2, stride=2
        feats1_avg = self.avgPool(data=feats1, kernelSize=2, stride=2)
        
        # CNN layer2: Kernel=[B,5,5]  stride=1
        feats2 = self.conv(data=feats1_avg, kernel=self.W_16x5x5, stride=1, padding=0)
        
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
    
    # Xavier initial
    def xavierInit(self, c_in, c_out, h, w, fc=False):
        fan_1 = c_out * w * h
        fan_2 = c_in * w * h
        ratio = np.sqrt(6.0 / (fan_1 + fan_2))
        if fc == True:
            params = ratio * (2*np.random.random((c_in, c_out, w, h)) - 1)
            params = params.reshape(c_in, c_out)
        else:
            params = ratio * (2*np.random.random((c_out, w, h)) - 1)
            
        return params
        
    def reLU(self, x):
        return np.where(x>0,x,0)
            
    def softmax(self, x):
        x_max = np.expand_dims(np.max(x, axis=1),1)
        e_x = np.exp(x-x_max)
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
                        if row < padding or row >= h or col < padding or col >= w : continue
                        
                        # otherwise, keep same
                        data_pad[b, c, row, col] = data[b, c, row, col]
        
        return data_pad

    # average pool
    def avgPool(self, data, kernelSize, stride):
        data_batch, data_chs, data_h, data_w = np.shape(data)
        k_h, k_w = kernelSize, kernelSize
        out_h = np.uint16((data_h - k_h) / stride) + 1
        out_w = np.uint16((data_w - k_w) / stride) + 1
        scalar_h = out_h/data_h
        scalar_w = out_w/data_w
        avgPool_mat = np.zeros((data_batch, data_chs, out_h, out_w), np.float32)
        offX = k_w//2
        offY = k_h//2
        
        for b in range(data_batch):
            for chs in range(data_chs):
                
                for row in range(data_h):
                    for col in range(data_w):
                        acc = 0.0
                        # accumlate all the element in data within Kernerl
                        for x in range(k_h):
                            for y in range(k_w):
                                data_col = col+x-offX
                                data_row = row+y-offY
                                # skip index out of bounds
                                if data_col < 0 or data_col >= data_w or data_row <0 or data_row >= data_h: continue
                                acc+= data[b, chs, data_row, data_col]
                        scaledRow = np.int16(row*scalar_h)
                        scaledCol = np.int16(col*scalar_w)
                        avgPool_mat[b,chs,scaledRow,scaledCol] = acc / (k_h*k_w)
            
        return avgPool_mat
        
    def conv(self, data, kernel, stride, padding):
        # padding data
        data_pad = self.paddingData(data, padding)
        
        #
        data_batch, data_chs, data_h, data_w = np.shape(data)
        
        k_chs, k_h, k_w = np.shape(kernel)
        out_h = np.uint16((data_h - k_h + 2*padding) / stride) + 1
        out_w = np.uint16((data_w - k_w + 2*padding) / stride) + 1
        
        # outputs
        conv_mat = np.zeros((data_batch, k_chs, out_h, out_w))
        
        for b in range(data_batch):
            for k_c in range(k_chs):
                for c in range(data_chs):
                    
                    if data_chs!=1 and self.FTtable[c, k_c] ==0:continue
                    
                    for row in range(out_h):
                        for col in range(out_w):
                            # accumlate all the element-wise multiple
                            for x in range(k_w):
                                for y in range(k_h):
                                    data_row = row+y-(k_h//2)
                                    data_col = col+x-(k_w//2)
                                    # skip index out of bounds
                                    if data_col < 0 or data_col >= out_w or data_row <0 or data_row > out_h: continue
                                    conv_mat[b, k_c, row, col]+= (data_pad[b,c,data_row,data_col]*kernel[k_c,x,y])
        
        return conv_mat
    
    # Convolve of BackPropagation
    def conv_bp(self, feats_data, det_data, kernel, stride, padding):
        
        ft_b, ft_chs, ft_h, ft_w = np.shape(feats_data)
        dt_b, dt_chs, dt_h, dt_w = np.shape(det_data)
        k_chs, k_h, k_w = np.shape(kernel)
        
        # padding data
        padding = ((ft_h +1)*stride+k_h -dt_h)//2
        dt_pad = self.paddingData(det_data, padding)
        dt_pad_h = dt_h+padding*2
        dt_pad_w = dt_w+padding*2

        # outputs
        det_W = np.zeros(np.shape(kernel))
        det_feats = np.zeros(np.shape(feats_data))
        
        # solve det feats
        for b in range(ft_b):
            for c in range(ft_chs):
                for k_c in range(k_chs):
                    
                    # check table: it is not accumlative if it is 0
                    if ft_chs!=1 and self.FTtable[c, k_c]==0:continue
                    
                    # convolve2D
                    for ft_row in range(ft_h):
                        for ft_col in range(ft_w):
                            
                            # sovle det feat
                            for x in range(k_w):
                                for y in range(k_h):
                                    dt_row = ft_row+y - (k_h//2)
                                    dt_col = ft_col+x - (k_w//2)
                                    if dt_row<0 or dt_row>=dt_pad_h or dt_col<0 or dt_col>=dt_pad_w:continue
                                    det_feats[b,c,ft_row,ft_col] += dt_pad[b,c,dt_row,dt_col]*kernel[k_c,x,y]
                            
                    # solve det W
                    for k_row in range(k_h):
                        for k_col in range(k_w):
                            for x in range(dt_w):
                                for y in range(dt_h):
                                    ft_row = k_row+y - (dt_h//2)
                                    ft_col = k_col+x - (dt_w//2)
                                    if ft_row<0 or ft_row>=ft_h or ft_col<0 or ft_col>=ft_w:continue
                                    det_W[k_c,k_row,k_col] += det_feats[b,c,ft_row,ft_col]*det_data[b,c,x,y]

        return det_feats, det_W    
                
    def train(self, x_train, y_train):

        #################### forward
        # W1
        a1 = self.conv(data=x_train, kernel=self.W_6x5x5, stride=1, padding=0)
        avgp1 = self.avgPool(data=a1, kernelSize=2, stride=2)
        
        # W2
        a2 = self.conv(data=avgp1, kernel=self.W_16x5x5, stride=1, padding=0)
        avgp2 = self.avgPool(data=a2, kernelSize=2, stride=2)
        
        # flatten features into series
        features_flatten = np.reshape(avgp2, (self.batchSize, 5*5*16))

        # FC3
        a3 = np.dot(features_flatten, self.W_nx120)
        sig3 = self.reLU(a3)
        
        # FC4
        a4 = np.dot(sig3, self.W_120x84) 
        sig4 = self.reLU(a4)
        
        # FC5
        a5 = np.dot(sig4, self.W_84xn)
        
        # Softmax
        y_predict = np.argmax(self.softmax(a5),axis=1)
        
        # cost
        probs = [self.softmax(a5)[i][y_predict[i]] for i in range(self.batchSize)]
        self.cost = - (1/self.batchSize)*np.sum(np.log(probs))
        print("Cost: %.4f" %(self.cost))
        ##################### back propagation
        
        # init Y
        Y = np.zeros((self.batchSize, self.classes))
        for b in range(self.batchSize):
            Y[b, y_train[b]] = 1.0
            
        y_softmax_out = self.softmax(a5)
        
        ############## Updates weights of fully connected layer
        # F5 dW_84xN
        det_l = y_softmax_out - Y
        dW_84xn = np.dot(np.transpose(det_l), (sig4))
        self.W_84xn -= np.transpose(dW_84xn) * self.learn_rate
        
        # F4 dW_120x84
        det_l = np.dot(self.W_84xn, np.transpose(det_l))*np.transpose(sig4)
        dW_120x84 = np.dot(det_l, sig3)
        self.W_120x84 -= np.transpose(dW_120x84) * self.learn_rate
        
        # F3 dW_nx120
        avgp2_reshape = np.reshape(avgp2, (self.batchSize, 5*5*16))
        det_l = np.dot(self.W_120x84, det_l)*np.transpose(sig3)
        dW_nx120 = np.dot(det_l, avgp2_reshape)
        self.W_nx120 -= np.transpose(dW_nx120) * self.learn_rate
        
        ############## Updates weights of features layer
        # W2 dW_16x5x5 
        # delt_avgp2
        det_l = np.dot(self.W_nx120, det_l)*np.transpose(avgp2_reshape)
        # bp_avgpooling -> a2
        det_avgp2 = np.reshape(det_l, (self.batchSize, 16,5,5))
        det_avgp2 = np.repeat(det_avgp2,2, axis=2)
        det_a2 = np.repeat(det_avgp2,2, axis=3)
        # bp_conv -> avgp1
        det_l, dw_16x5x5 = self.conv_bp(feats_data=avgp1, det_data=det_a2, kernel=self.W_16x5x5,stride=1,padding=0)
        self.W_16x5x5 -= dw_16x5x5*self.learn_rate
        # bp_avgpooling -> a1
        det_avgp1 = np.repeat(det_l,2, axis=2)
        det_a1 = np.repeat(det_avgp1,2, axis=3)
        # bp_cov -> W1 dW_6x5x5
        det_l, dw_6x5x5 = self.conv_bp(feats_data=x_train, det_data=det_a1, kernel=self.W_6x5x5, stride=1,padding=0)
        # W1 dW_6x5x5
        self.W_6x5x5 -= dw_6x5x5
        
        print("trained a batch")
    
        
        
    def predict(self, x):
        features = self.CNN_features_layer(x)
        # flatten features into series
        features_flatten = np.reshape(features, (self.batchSize, 5*5*16))
        y_predict = self.FC_classifier_layer(features_flatten)
        y_labels = np.argmax(y_predict, axis=1)
        
        return y_labels
        