import os
import torch
import random
from network.HSPose import HSPose 
from tools.geom_utils import generate_RT
from config.config import *
from absl import app
# absl.flags库允许开发者在脚本执行前通过命令行方便地配置参数，而无需修改脚本源代码。
# 这些参数通过FLAGS.参数名的方式引用，在脚本内部直接使用这些变量即可访问和应用传入的值，
# 实现了命令行参数到脚本内部变量的绑定和使用。
FLAGS = flags.FLAGS


# python -m evaluation.evaluate  
# --model_save output/models/HS-Pose_weights/eval_result 
# --resume 1 
# --resume_model ./output/models/HS-Pose_weights/model.pth 
# --eval_seed 1677483078

# 
# --eval_seed: 用于设置随机种子的参数。
# --data_dir: 数据集的路径。
# --batch_size: 训练或评估时的批量大小。
# --learning_rate: 学习率，影响模型训练的速度和质量。
# --model_dir: 模型保存或加载的目录。
# --num_epochs: 训练轮次。
# --gpu: 是否使用GPU或指定哪个GPU进行训练。
# --log_level: 控制日志输出的详细程度（如DEBUG, INFO, WARNING等）。
# --model_save=/path/to/save/model.pth来指定模型的保存位置。

from evaluation.load_data_eval import PoseDataset
import numpy as np
import time

# from creating log
import tensorflow as tf
from evaluation.eval_utils import setup_logger
from evaluation.eval_utils_v1 import compute_degree_cm_mAP
from tqdm import tqdm

# 生成随机数
def seed_init_fn(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


    # np.random.seed(seed): 这一行是用来设定NumPy库中的随机数生成器的种子。
    # NumPy是一个广泛应用于科学计算的Python库，尤其是在处理数组和矩阵时。
    # 通过设定种子，可以确保每次执行时生成的随机数序列相同。
    # random.seed(seed): 这一行则是针对Python标准库中的random模块设置种子。
    # 这个模块提供了各种随机数生成功能，设置种子可以保证每次程序运行时这些随机数的生成顺序和值保持一致。
    # torch.manual_seed(seed): 这一行是针对PyTorch深度学习框架设置随机种子。
    # PyTorch是一个广泛用于机器学习和深度学习任务的库。
    # 通过torch.manual_seed，可以确保所有CPU上的PyTorch操作（比如权重初始化、数据洗牌等）都有确定性的行为。

    return

# 设置全局变量，用显卡设备
device = 'cuda'

def evaluate(argv):
    # 如果没用随机数就根据时间生成一个
    # FLAGS: 这是一个命令行标志对象的引用，通常用于存储用户通过命令行传入的参数。
    # eval_seed是其中的一个参数，用户可以设置该参数来决定是否以及如何设定随机种子。
    if FLAGS.eval_seed == -1:
        seed = int(time.time())
    else:
        # eval_seed就是这个随机数的数值
        seed = FLAGS.eval_seed
    # 带哦用上面的生成随机数函数seed_init_fn
    seed_init_fn(seed)


    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)

    # 禁用 eager execution（即时执行模式）
    tf.compat.v1.disable_eager_execution()
    # 日志操作
    logger = setup_logger('eval_log', os.path.join(FLAGS.model_save, 'log_eval.txt'))
    # 模式赋值 
    Train_stage = 'PoseNet_only'
    # 设置为非训练模式
    FLAGS.train = False
    
    # 处理文件路径并从中提取模型名称
    # os.path.basename:取出路径中文件名部分
    # .split('.'): 这个操作将文件名按.分隔成一个列表。在大多数情况下，这用于分离文件的基本名称和扩展名。
    # 例如，对于model.pth，.split('.')会得到['model', 'pth']。
    # [0]: 最后，通过索引[0]选取分割后列表的第一个元素，即去掉扩展名的文件基本名称。
    # 所以，如果FLAGS.resume_model指向的文件名为model.pth，经过上述操作后，model_name就会被赋值为model。
    model_name = os.path.basename(FLAGS.resume_model).split('.')[0]
    # build dataset annd dataloader

    # 模式设置为测试，在这里初始化传入参数，
    val_dataset = PoseDataset(source=FLAGS.dataset, mode='test')

    # 设置输出路径
    output_path = os.path.join(FLAGS.model_save, f'eval_result_{model_name}')
    # f'eval_result_{model_name}'：这是Python的f-string（格式化字符串字面量），用于插入变量表达式并格式化字符串。
    # 在这里，model_name变量是从模型文件名中提取的
    # path = os.path.join("mydir", "subdir", "file.txt")
    # 结果是：mydir/subdir/file.txt
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #如果目录不存在，就创建目录
    
    import pickle

    # 记录时间，图像数量记录
    t_inference = 0.0
    img_count = 0
    # 连接数据参数路径
    pred_result_save_path = os.path.join(output_path, 'pred_result.pkl')
    # 如果存在这个模型
    if os.path.exists(pred_result_save_path):
        # as 是为了给被打开的文件起一个别名 with是打开文件建华异常处理。不论是否异常退出，都可以关闭文件。
        with open(pred_result_save_path, 'rb') as file:
            pred_results = pickle.load(file)
        img_count = 1
    else:
    # 如果数据不存在，就需要构建网络模型
        # Train_stage = 'PoseNet_only' Train_stage是一个模式，构建了一个网络
        network = HSPose(Train_stage)
        network = network.to(device)

        # 当resume为true时，继续前一次的训练过程。
        if FLAGS.resume: 
            # ['posenet_state_dict']这部分是用来提取这个字典中特定部分的数据先前已经训练好的 PoseNet 的所有参数（即权重和偏置）。
            # 通过调用 torch.load() 加载保存好的模型文件，再用 'posenet_state_dict' 来索引出来，这样就可以方便的获取和管理这些参数了。
            state_dict = torch.load(FLAGS.resume_model)['posenet_state_dict']
            unnecessary_nets = ['posenet.face_recon.conv1d_block', 'posenet.face_recon.face_head', 'posenet.face_recon.recon_head']
            # 这里的逻辑是为了移除一些不必要的网络层。
            for key in list(state_dict.keys()):
                for net_to_delete in unnecessary_nets:
                    if key.startswith(net_to_delete):
                        state_dict.pop(key)
                # Adapt weight name to match old code version. 
                # Not necessary for weights trained using newest code. 
                # Dose not change any function. 
                if 'resconv' in key:
                    state_dict[key.replace("resconv", "STE_layer")] = state_dict.pop(key)
            network.load_state_dict(state_dict, strict=True) 
        else:
            # 在Python中，raise NotImplementedError 是一种异常抛出机制，用于标记某个方法或功能尚未在类或模块中实现。
            # 当程序执行到包含 raise NotImplementedError 的代码时，会立即停止当前函数或方法的执行，并抛出一个 NotImplementedError 异常，通常附带一个可选的消息说明为何功能未实现。
            raise NotImplementedError
        # start to test
        # eval()将神经网络设置为评估模式，这通常会关闭如Dropout和Batch Normalization层的训练时行为，以得到确定性的预测结果。
        network = network.eval()
        pred_results = []
        # 遍历验证数据集(val_dataset)
        # tqdm 添加进度条!!!，提供进度的实时可视化。
        for i, data in tqdm(enumerate(val_dataset, 1), dynamic_ncols=True):
            # enumerate函数用于遍历序列（在这里是val_dataset）,同时提供每个元素的索引（从0开始）。i 的值会从1开始，并随着循环的迭代递增
            # 通过传递1作为第二个参数，使索引从1开始计数，这样在显示进度时更加自然（通常人们倾向于从1开始计数而不是0）。每次循环时：
            # data 就是val_dataset中的下一个数据样本。首次循环时是第一个样本，第二次循环是第二个样本，依此类推，直到数据集中的所有样本都被遍历完。
            if data is None:
                continue
            # print(data)
            # gts:ground truths真实数据。  
            data, detection_dict, gts = data
            mean_shape = data['mean_shape'].to(device)
            sym = data['sym_info'].to(device)
            # 如果数据长度为零，那么把旋转矩阵也置零
            if len(data['cat_id_0base']) == 0:
                # 预测的旋转矩阵（Rotation Matrices）的集合
                detection_dict['pred_RTs'] = np.zeros((0, 4, 4))
                # pred_scales是模型预测的物体在三维空间中的相对尺寸或缩放比例，用以补充纯几何变换（旋转和平移位）之外的信息，完整描述物体的三维状态。
                detection_dict['pred_scales'] = np.zeros((0, 4, 4))
                pred_results.append(detection_dict)
                continue
            t_start = time.time()
            output_dict \
                = network(
                          PC=data['pcl_in'].to(device), 
                          obj_id=data['cat_id_0base'].to(device), 
                          mean_shape=mean_shape,
                          sym=sym,
                        #   def_mask=data['roi_mask'].to(device)
                          )
            p_green_R_vec = output_dict['p_green_R'].detach()
            p_red_R_vec = output_dict['p_red_R'].detach()
            p_T = output_dict['Pred_T'].detach()
            p_s = output_dict['Pred_s'].detach()
            f_green_R = output_dict['f_green_R'].detach()
            f_red_R = output_dict['f_red_R'].detach()
            pred_s = p_s + mean_shape
            pred_RT = generate_RT([p_green_R_vec, p_red_R_vec], [f_green_R, f_red_R], p_T, mode='vec', sym=sym)

            t_inference += time.time() - t_start
            img_count += 1

            if pred_RT is not None:
                pred_RT = pred_RT.detach().cpu().numpy()
                pred_s = pred_s.detach().cpu().numpy()
                detection_dict['pred_RTs'] = pred_RT
                detection_dict['pred_scales'] = pred_s
            else:
                assert NotImplementedError
            pred_results.append(detection_dict)
            torch.cuda.empty_cache()
        with open(pred_result_save_path, 'wb') as file:
            pickle.dump(pred_results, file)
        print('inference time:', t_inference / img_count)
    if FLAGS.eval_inference_only:
        import sys
        sys.exit()
        # 这段代码的意思是，如果FLAGS.eval_inference_only这个条件为True，则通过Python的sys模块调用exit()函数来终止当前脚本的执行。


    # 这行代码的意思是生成一个列表，包含了从0到60之间所有整数（包括0和60），步长为1。
    degree_thres_list = list(range(0, 61, 1))
    # 生成0到100（包含100）再每个元素除以2
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]

    # iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, output_path, degree_thres_list, shift_thres_list,
    #                                                  iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True,)
    synset_names = ['BG'] + ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    if FLAGS.per_obj in synset_names:
        idx = synset_names.index(FLAGS.per_obj)
    else:
        idx = -1
    
    # 在这里进行位姿估计任务的评估过程。
    iou_aps, pose_aps = compute_degree_cm_mAP(pred_results, synset_names, output_path, degree_thres_list,
                                              shift_thres_list,
                                              iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True, )

    # # fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    shift_10_idx = shift_thres_list.index(10)

    messages = []

    if FLAGS.per_obj in synset_names:
        messages.append('Evaluation Seed: {}'.format(seed))
        messages.append('mAP:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
        messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
        messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
        messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
        messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))
        messages.append('5 degree: {:.1f}'.format(pose_aps[idx, degree_05_idx, -1] * 100))
        messages.append('10 degree: {:.1f}'.format(pose_aps[idx, degree_10_idx, -1] * 100))
        messages.append('2cm: {:.1f}'.format(pose_aps[idx, -1, shift_02_idx] * 100))
        messages.append('5cm: {:.1f}'.format(pose_aps[idx, -1, shift_05_idx] * 100))
        messages.append('10cm: {:.1f}'.format(pose_aps[idx, -1, shift_10_idx] * 100))
        messages.append("Inference time: {:06f}  Average: {:06f}/image".format(t_inference, t_inference / img_count))
    else:
        messages.append('Evaluation Seed: {}'.format(seed))
        messages.append('average mAP:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
        messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
        messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
        messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
        messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))
        messages.append('5 degree: {:.1f}'.format(pose_aps[idx, degree_05_idx, -1] * 100))
        messages.append('10 degree: {:.1f}'.format(pose_aps[idx, degree_10_idx, -1] * 100))
        messages.append('2cm: {:.1f}'.format(pose_aps[idx, -1, shift_02_idx] * 100))
        messages.append('5cm: {:.1f}'.format(pose_aps[idx, -1, shift_05_idx] * 100))
        messages.append('10cm: {:.1f}'.format(pose_aps[idx, -1, shift_10_idx] * 100))
        messages.append("Inference time: {:06f}  Average: {:06f}/image".format(t_inference, t_inference / img_count))

        for idx in range(1, len(synset_names)):
            messages.append('category {}'.format(synset_names[idx]))
            messages.append('mAP:')
            messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
            messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
            messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
            messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
            messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
            messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
            messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
            messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))
            messages.append('5 degree: {:.1f}'.format(pose_aps[idx, degree_05_idx, -1] * 100))
            messages.append('10 degree: {:.1f}'.format(pose_aps[idx, degree_10_idx, -1] * 100))
            messages.append('2cm: {:.1f}'.format(pose_aps[idx, -1, shift_02_idx] * 100))
            messages.append('5cm: {:.1f}'.format(pose_aps[idx, -1, shift_05_idx] * 100))
            messages.append('10cm: {:.1f}'.format(pose_aps[idx, -1, shift_10_idx] * 100))

    for msg in messages:
        logger.info(msg)


if __name__ == "__main__":
    app.run(evaluate)
    # absl.app模块特别用于处理Python应用程序的命令行接口和程序入口点。
    # 它提供了一些高级功能，比如命令行参数解析（类似于argparse）、模块初始化、程序生命周期管理（比如优雅地处理信号和程序退出）等，
    # 使得编写可配置的、支持多种运行模式的命令行应用程序变得更加简单和规范。
    # 常见的使用场景包括定义主函数（通常命名为main），
    # 并使用absl.app.run(main)作为程序的入口点，这样可以自动处理命令行参数解析和程序的启动与结束流程。
    # 此外，它也支持定义多个子命令，使得大型应用可以组织得更加模块化和易于管理。
