import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer


def test():
    import os
    im_file = '/home/zli/WorkSpace/PyWork/TorchWorkSpace/faster_rcnn_pytorch-master/demo/emp_test_image/croped/2017-04-27__13-11-28-998_4_0.jpg'
    out_file = im_file[:-4] + '_out.jpg'
    # im_file = 'data/VOCdevkit2007/VOC2007/JPEGImages/009036.jpg'
    # im_file = '/media/longc/Data/data/2DMOT2015/test/ETH-Crossing/img1/000100.jpg'
    image = cv2.imread(im_file)

    model_file = '/media/zli/data/VOC/models/saved_model_emp/faster_rcnn_100000.h5'#'/media/zli/data/Terra/R-CNN/VGGnet_fast_rcnn_iter_70000.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch3/faster_rcnn_100000.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch2/faster_rcnn_2000.h5'
    detector = FasterRCNN()
    network.load_net(model_file, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')

    # network.save_net(r'/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5', detector)
    # print('save model succ')

    t = Timer()
    t.tic()
    # image = np.zeros(shape=[600, 800, 3], dtype=np.uint8) + 255
    dets, scores, classes = detector.detect(image, 0.7)
    runtime = t.toc()
    print('total spend: {}s'.format(runtime))

    im2show = np.copy(image)
    for i, det in enumerate(dets):
        det = tuple(int(x) for x in det)
        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
        cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
    cv2.imwrite(os.path.join(out_file), im2show)
    #cv2.imshow('demo', im2show)
    #cv2.waitKey(0)

def test_dir(in_dir, out_dir):
    
    import os
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    model_file = '/media/zli/data/VOC/models/saved_model_3dPanicle/faster_rcnn_100000.h5'
    detector = FasterRCNN()
    network.load_net(model_file, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')
    
    csv_out_file = os.path.join(out_dir, 'panicle_counts.csv')
    csv_handle = open(csv_out_file, 'w')
    
    list_files = os.walk(in_dir)
    for root, dirs, files in list_files:
        for f in files:
            img_path = os.path.join(in_dir, f)
            out_file = os.path.join(out_dir, f)
            
            image = cv2.imread(img_path)

            t = Timer()
            t.tic()
            # image = np.zeros(shape=[600, 800, 3], dtype=np.uint8) + 255
            dets, scores, classes = detector.detect(image, 0.95)
            runtime = t.toc()
            print('total spend: {}s'.format(runtime))
        
            im2show = np.copy(image)
            for i, det in enumerate(dets):
                det = tuple(int(x) for x in det)
                cv2.rectangle(im2show, det[0:2], det[2:4], (0, 85, 255), 1)
                cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (0, 0, 255), thickness=1)
            cv2.imwrite(os.path.join(out_file), im2show)
            
            out_line = '{},{}\n'.format(f, len(dets))
            csv_handle.write(out_line)
    
    csv_handle.close()
    return

if __name__ == '__main__':
    #test()
    in_dir = '/media/zli/Elements/terra_evaluation/panicle_evaluation/panicle_for_evaluation'
    out_dir = '/media/zli/Elements/terra_evaluation/panicle_evaluation/pipeline_outputs'
    test_dir(in_dir, out_dir)
    
    