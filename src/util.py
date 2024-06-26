import numpy as np
import math
import cv2
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        if len(weights_name.split('.'))>4:  # body25
            transfered_model_weights[weights_name] = model_weights['.'.join(
                weights_name.split('.')[3:])]
        else:
            transfered_model_weights[weights_name] = model_weights['.'.join(
                weights_name.split('.')[1:])]
    return transfered_model_weights

# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset, model_type='body25'):
    stickwidth = 4
    if model_type == 'body25':
        limbSeq = [[1,0],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],\
                [10,11],[8,12],[12,13],[13,14],[0,15],[0,16],[15,17],[16,18],\
                [11,24],[11,22],[14,21],[14,19],[22,23],[19,20]]
        njoint = 25
    else:
        limbSeq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], \
                    [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], \
                    [0, 15], [15, 17], [2, 16], [5, 17]]
        njoint = 18

    # colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
    #           [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
    #           [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255,255,0], [255,255,85], [255,255,170],\
                [255,255,255],[170,255,255],[85,255,255],[0,255,255]]

    for i in range(njoint):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(njoint-1):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            # print('original (mX,mY,length,angle)',(mX,mY,length,angle))
            # print(f'original cv2.ellipse2Poly((int({mY}), int({mX})), (int({length} / 2), {stickwidth}), int({angle}), 0, 360, 1)')
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            # print(f'cv2.fillConvexPoly(cur_canvas, polygon, colors[i])')
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas
#subsets [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, -1.0, 11.0, 12.0, -1.0, 13.0, 14.0, 15.0, 16.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 26.650803712300775, 17.0]]
#candidates [[983.0, 172.0, 0.8991263508796692, 0.0], [980.0, 352.0, 0.930037796497345, 1.0], [848.0, 342.0, 0.8652207255363464, 2.0], [811.0, 598.0, 0.8107873797416687, 3.0], [806.0, 817.0, 0.7464589476585388, 4.0], [1120.0, 361.0, 0.8538270592689514, 5.0], [1148.0, 601.0, 0.6797391176223755, 6.0], [1149.0, 834.0, 0.5189468264579773, 7.0], [968.0, 757.0, 0.6468111276626587, 8.0], [876.0, 756.0, 0.6387956142425537, 9.0], [854.0, 1072.0, 0.4211728572845459, 10.0], [1057.0, 759.0, 0.6311940550804138, 11.0], [1038.0, 1072.0, 0.38531172275543213, 12.0], [955.0, 146.0, 0.925083339214325, 13.0], [1016.0, 151.0, 0.9023998379707336, 14.0], [909.0, 167.0, 0.9096773862838745, 15.0], [1057.0, 173.0, 0.8605436086654663, 16.0]]
def  get_bodypose(candidate, subset, model_type='coco'):
    stickwidth = 4
    if model_type == 'body25':
        limbSeq = [[1,0],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],\
                [10,11],[8,12],[12,13],[13,14],[0,15],[0,16],[15,17],[16,18],\
                [11,24],[11,22],[14,21],[14,19],[22,23],[19,20]]
        njoint = 25
    else:
        limbSeq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], \
                    [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], \
                    [0, 15], [15, 17], [2, 16], [5, 17]]
        njoint = 18

    # colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
    #           [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
    #           [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255,255,0], [255,255,85], [255,255,170],\
                [255,255,255],[170,255,255],[85,255,255],[0,255,255]]

    x_y_circles=[]
    for i in range(njoint):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2] # 983.0, 172.0
            x_y_circles.append((x, y))
            # cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    x_y_sticks=[]
    for i in range(njoint-1):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])] #0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, -1.0, 11.0, 12.0, -1.0, 13.0, 14.0, 15.0, 16.0, -1.0, -1.0, -1.0, -1.0, -1.0
            if -1 in index:
                continue
            # cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            x_y_sticks.append((mY, mX,angle,length))
            # print('new  (mX,mY,length,angle)',(mX,mY,length,angle))
            # polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            # cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            # canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return (x_y_circles,x_y_sticks,)

#all_hands_peaks[[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1100, 858], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [858, 859], [868, 894], [873, 938], [0, 0], [802, 920], [807, 961], [821, 977], [836, 992], [0, 0], [781, 955], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]
def draw_handpose(canvas, all_hand_peaks, show_number=False):
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    fig = Figure(figsize=plt.figaspect(canvas))

    fig.subplots_adjust(0, 0, 1, 1)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    bg = FigureCanvas(fig)
    ax = fig.subplots()
    ax.axis('off')
    ax.imshow(canvas)

    width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()

    for peaks in all_hand_peaks:
        for ie, e in enumerate(edges):
            if np.sum(np.all(peaks[e], axis=1)==0)==0:
                x1, y1 = peaks[e[0]]
                x2, y2 = peaks[e[1]]
                # print(f'original ax.plot([{x1}, {x2}], [{y1}, {y2}], color=matplotlib.colors.hsv_to_rgb([ie/float({len(edges)}), 1.0, 1.0]))')
                ax.plot([x1, x2], [y1, y2], color=matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0]))

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            # print(f"original ax.plot({x}, {y}, 'r.')")
            ax.plot(x, y, 'r.')
            if show_number:
                ax.text(x, y, str(i))
    # print(f'width = {width}, height={height}')
    bg.draw()
    canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return canvas

def get_handpose(all_hand_peaks, show_number=False):
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    # fig = Figure(figsize=plt.figaspect(canvas))

    # fig.subplots_adjust(0, 0, 1, 1)
    # fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    # bg = FigureCanvas(fig)
    # ax = fig.subplots()
    # ax.axis('off')
    # ax.imshow(canvas)

    # width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()
    export_edges=[[],[]]
    export_peaks=[[],[]]
    for idx,peaks in enumerate(all_hand_peaks):
        for ie, e in enumerate(edges):
            if np.sum(np.all(peaks[e], axis=1)==0)==0:
                x1, y1 = peaks[e[0]]
                x2, y2 = peaks[e[1]]
                export_edges[idx].append((ie,(x1, y1),(x2, y2)))
                # ax.plot([x1, x2], [y1, y2], color=matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0]))

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            # ax.plot(x, y, 'r.')
            # if show_number:
            #     ax.text(x, y, str(i))

            export_peaks[idx].append((x,y,str(i)))
    # bg.draw()
    # canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return (export_edges,export_peaks)

# image drawed by opencv is not good.
def draw_handpose_by_opencv(canvas, peaks, show_number=False):
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    for ie, e in enumerate(edges):
        if np.sum(np.all(peaks[e], axis=1)==0)==0:
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0])*255, thickness=2)

    for i, keyponit in enumerate(peaks):
        x, y = keyponit
        cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
        if show_number:
            cv2.putText(canvas, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), lineType=cv2.LINE_AA)
    return canvas

# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def handDetect(candidate, subset, oriImg):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    ratioWristElbow = 0.33
    detect_result = []
    
    image_height, image_width = oriImg.shape[0:2]
    #print(f'handDetect ---------- {image_height}, {image_width}')
    for person in subset.astype(int):
        # if any of three not detected
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue
        hands = []
        #left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[[2, 3, 4]]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
            # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
            # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
            # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
            # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
            # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # x-y refers to the center --> offset to topLeft point
            # handRectangle.x -= handRectangle.width / 2.f;
            # handRectangle.y -= handRectangle.height / 2.f;
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image
            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > image_width: width1 = image_width - x
            if y + width > image_height: width2 = image_height - y
            width = min(width1, width2)
            # the max hand box value is 20 pixels
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

    '''
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    '''
    return detect_result

def drawStickmodel(oriImg,x_ytupple,x_y_sticks,export_edges,export_peaks):
    canvas = copy.deepcopy(oriImg)


    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], 
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], 
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255,255,0], [255,255,85], [255,255,170],
                [255,255,255],[170,255,255],[85,255,255],[0,255,255]]
    stickwidth=4

    for idx,(mX,mY,angle,length) in enumerate(x_y_sticks):
        cur_canvas = canvas.copy()
        # print(f'new cv2.ellipse2Poly((int({mY}), int({mX})), (int({length} / 2), {stickwidth}), int({angle}), 0, 360, 1)')
        polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[idx])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)



    for idx,(x,y) in enumerate(x_ytupple):
        cv2.circle(canvas, (int(x), int(y)), 4, colors[idx], thickness=-1)
        

    ## Handpose
    fig = Figure(figsize=plt.figaspect(canvas))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    bg = FigureCanvas(fig)
    ax = fig.subplots()
    ax.axis('off')
    ax.imshow(canvas)

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
                [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for both_hand_edges in export_edges:
        for (ie,(x1, y1),(x2, y2)) in both_hand_edges:
            # print(f'new ax.plot([{x1}, {x2}], [{y1}, {y2}], color=matplotlib.colors.hsv_to_rgb([ie/float({len(edges)}), 1.0, 1.0]))')
            ax.plot([x1, x2], [y1, y2], color=matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0]))

    width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()

    for both_hand_peaks in export_peaks:
        for (x,y,text) in both_hand_peaks:
            # print(f"new ax.plot({x}, {y}, 'r.')")
            ax.plot(x, y, 'r.')



    # print(f'NEW width = {width}, height={height}')
    bg.draw()

    canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    ####


    # cv2.imwrite('C:/Users/spsar/Downloads/MVI_5177.MOV-transformed/MVI_5177.MOV-GaussianBlur/MVI_5177.MOV-14-modified.jpg', canvas) 
    return canvas

def crop_to_drawing(image):
  """
  Crops an image to the tight bounding rectangle of non-zero pixels.

  Args:
      image: A NumPy array representing the image.

  Returns:
      A cropped image (NumPy array) containing only the drawing area.
  """
  image=np.transpose(image, (2, 0, 1))
  united_x,united_h=0,0
  for channel in np.arange(image.shape[0]):
    x, y, w, h = cv2.boundingRect(image[channel])
    if x>united_x:
        united_x=x

    if h>united_h:
        united_h=h

  for channel in np.arange(image.shape[0]):
    # Crop the image
    image[channel] = image[channel][y:y+united_h, x:x+united_x]
  return image.transpose(image, (1,2,0))

# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j
