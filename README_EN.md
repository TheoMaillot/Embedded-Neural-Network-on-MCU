# Deployment of a Deep Neural Network on a Microcontroller

The goal of this project is to deploy a pre-trained deep neural network on the CIFAR-10 dataset for an image classification application. The main objective is to adapt this model so that it can be embedded on a microcontroller while meeting the inherent constraints of embedded systems: storage capacity, limited computing power, etc. Moreover, security is a domain that affects everyone, including AI. This is why, at the end of the project, attacks will be carried out on the AI models to observe their parameters and functions, and also to understand that attacking AIs to trick them is actually quite simple.

## Prerequisites

- Up-to-date **Python**
- **STM32CubeIDE** installed (with **X-CUBE-AI** pack)  
- An **STM32L4R9 Discovery Kit**
- Required Python libraries (numpy, pyserial, etc.)  

## Installation

```bash
gh repo clone TheoMaillot/Embedded-Neural-Network-on-MCU
```

## Usage

1. Decompress `dataset.rar` to extract the two `.npy` files  
2. Open `serial_evaluation.py` and set your own paths to the `.NPY` files at [lines 87 and 88](https://github.com/TheoMaillot/Embedded-Neural-Network-on-MCU/blob/main/serial_evaluation.py#L87-88)  
3. Plug in the board and specify your COM port at [line 5](https://github.com/TheoMaillot/Embedded-Neural-Network-on-MCU/blob/main/serial_evaluation.py#L5)  
4. Decompress [IA_Embeded.rar](./STM32cubeide-project/IA_Embeded.rar) and open the project with CubeIDE  
5. In X-CUBE-AI under *cifar10*, add your `.h5` file in *Model*, and your two `.npy` files in *Validation inputs* and *Validation outputs*  
6. Press **Run**  
7. Execute `serial_evaluation.py`  

---

## 1. Analysis of the existing model

The proposed model is based on VGG11 but adapted to 32×32 CIFAR-10 images, with a progressive architecture (32→64→128 filters) that reduces its size to 16 MB compared to 528 MB for the classic VGG11. It consists of 3×3 convolutions with ReLU, Batch Normalization to stabilize training, and regularization using SpatialDropout2D (0.25) in convolution blocks and Dropout (0.3) in dense layers (1024→512→10 neurons).

We tested this model before modifying it, achieving 83% accuracy and 53% loss. The performance is good, and we attempted to preserve it as much as possible while optimizing the model's size.

---

## 2. Study of the target microcontroller

For this project, we used the STM32L4R9 Discovery Kit. The key characteristics to consider when choosing the board are Flash memory, where the model will be stored, and RAM, which performs model computations.

According to the datasheet, the board provides:  
- **2 MB Flash**  
- **640 KB RAM**

The RAM should be sufficient, even if inference might take several seconds per image. However, the Flash is insufficient given the current model size (16 MB).

---

## 3. Evaluation of the embeddability of the initial model

To check the model's embeddability, we created a CubeIDE project for the board and analyzed our model using STM32CubeAI. Here are the results:

![Base model analysis](./img/first_analyse.png)

As expected, Flash memory is insufficient to contain the model: it requires 5.14 MB while the board only has 2 MB.  
On the other hand, RAM usage is only 148 KB without modifications, which is well below the available RAM.

---

## 4. Solutions to make the model embeddable

### a. Model compression

A first solution is to compress the model. STM32CubeAI directly enables this, offering three compression modes: low, medium, and high. Here are the results for the *high* mode:

![Compressed model analysis](./img/compressed_model.png)

After compression, the model size is reduced enough to fit into Flash memory, making it embeddable. This also works for *medium* compression, but not for *low*.

### b. Creating a lighter model

The second solution is to create a completely new model. We modified the `train.py` file to reduce the model size.

We analyzed the parts consuming the most memory. In this case, the two last dense layers (the output must remain at 10 neurons) took the most space. Reducing these dense layers to 256 and 64 allowed us to keep similar performance (0.87 accuracy) while reducing the model size to 1.7 MB.

---

## 5. Integration in an embedded project

The final development step consists of integrating the model into the board. We retrieved our `.h5` file and implemented it using CubeIDE. Then, the model must be executed on the board.

### a. Python side: sending data to the board and evaluation

The script `serial_evaluation.py` manages the serial communication between the PC and the board: it synchronizes UART, sends inputs (X_test) as float32, reads outputs from the STM32, compares predictions to labels (Y_test), and computes accuracy over a given number of iterations.

During early tests, we noticed that nothing was received from the board (*ValueError: attempt to get argmax of an empty sequence*). We added a `time.sleep(5)` at [line 75](https://github.com/TheoMaillot/Embedded-Neural-Network-on-MCU/blob/main/serial_evaluation.py#L75) after sending the inputs to give the board enough computation time. This is due to the STM32L4R9 RAM size, which slows computations compared to a PC.

---

### b. Embedded side: receiving data and inference

Adding STM32CubeAI generates many files, including `app_x-cube-ai.c`. It uses the STM32CubeAI framework to execute our embedded model. We completed the program to handle UART communication to receive images from Python, process them with the neural network, and return the 10-class probabilities via UART.

---

## 6. Performance evaluation on target

Running `serial_evaluation.py` gives the following results:

![Training results](./img/resultat_final.png)

### Performance
- **Accuracy**: 87% over 100 iterations  
- **Inference time**: ~6 seconds per image (10 minutes for 100 inferences)

### Analysis
- Accuracy is comparable to PC performance (83%)
- Inference time is long due to:
  - Limited computing power
  - UART communication delays

These results demonstrate that our optimized model runs effectively on the target.

---

## 7. Adversarial attacks with Projected Gradient Descent (PGD)

### Description

This section presents the attacks implemented in `adversarial_example.ipynb` using `projected_gradient_descent()` with two norms: **L2** and **L∞**. These attacks deceive image classification models, targeted or untargeted.

### a. Norm principles

| Norm | Characteristics | Usage |
|------|----------------|--------|
| **L∞** | Limits the maximum pixel perturbation (uniform modification up to ±ε) | Discrete and uniform attacks |
| **L2** | Limits total Euclidean norm (some pixels may have strong changes) | Concentrated pixel attacks |

---

### b. Untargeted attacks

Untargeted attacks aim to fool the model without controlling the output class.

#### L∞ norm

##### Base configuration

```python
adv_perturbation_budget = 0.05
pgd_iterations = 40
pgd_step_budget = 0.01
```

**Results:**  
- 10 out of 10 images fooled  
- Slight stripes on 32×32 images, but attack remains subtle

![Image_1_attack_normal](./img/Image_1_attack_normal.png)

##### Ineffective configuration (step too large)

If the step equals epsilon, success rate drops drastically as gradients bounce at the boundaries.

![Image_2_attack_normal_adv_001](./img/Image_2_attack_normal_adv_001.png)

##### Excessive epsilon

If ε is too large, the image becomes a chaotic color cloud:

![Image_8_attack_normal_Linf_eps_3_stp_01_ite_40](./img/Image_8_attack_normal_Linf_eps_3_stp_01_ite_40.png)

> **Note:** ε = 3 is excessively high for normalized images.

---

#### L2 norm

##### Principle

L2 allows large modifications on a few pixels, requiring a larger epsilon than L∞.

##### Visibility by resolution

| Resolution | Visibility | Explanation |
|-----------|------------|-------------|
| 32×32     | Visible    | Modified pixels stand out |
| 2048×2048 | Invisible  | Changes imperceptible |

##### Configuration 1: Moderate epsilon

```python
adv_perturbation_budget = 1.2
pgd_step_budget = 0.05
```

Result: pixels are changed but appear natural; prediction is wrong.

![Image_3_attack_normal_L2_eps_120_stp_005](./img/Image_3_attack_normal_L2_eps_120_stp_005.png)

##### Configuration 2: High epsilon

```python
adv_perturbation_budget = 3
pgd_step_budget = 0.05
pgd_iterations = 75
```

![Image_4_attack_normal_L2_eps_3_stp_005_ite_75](./img/Image_4_attack_normal_L2_eps_3_stp_005_ite_75.png)

---

### c. Targeted attacks

Targeted attacks force the model to predict a specific class (here: class 3).

#### Implementation

1. Set target class:

```python
target_class = 3
```

2. Modify function call:

```python
projected_gradient_descent(..., targeted=True, target_label=target_class)
```

#### Results

##### L∞ norm

![Image_6_attack_cible_Linf_eps_3_stp_01_ite_40](./img/Image_6_attack_cible_Linf_eps_3_stp_01_ite_40.png)

##### L2 norm

![Image_5_attack_cible_L2_eps_3_stp_005_ite_75](./img/Image_5_attack_cible_L2_eps_3_stp_005_ite_75.png)

---

### d. Summary of attacks

| Attack type | Norm | Optimal epsilon | Advantages | Drawbacks |
|-------------|------|----------------|------------|-----------|
| Untargeted | L∞ | Low (0.05) | Subtle | Requires very low epsilon |
| Untargeted | L2 | High (1.2–3) | Effective on large images | Visible on small images |
| Targeted | L∞ | Low–medium | Class control | Harder convergence |
| Targeted | L2 | High | Class control | Visible artifacts |

---

### Bit Flip Attacks

Unlike adversarial attacks on input data, bit flip attacks target the model weights stored in memory. By flipping one or more bits (fault injection, electromagnetic perturbation), an attacker can drastically degrade model performance or force specific mispredictions.

---

## 8. Conclusion

This project demonstrated the complete deployment pipeline of a neural network on an STM32L4R9 microcontroller.

### Main achievements
- Reduction of a CNN from 5 MB to 1.7 MB while maintaining 87% accuracy  
- Successful microcontroller integration respecting hardware constraints  
- UART communication between PC and STM32  
- Exploration of vulnerabilities via adversarial attacks  

### Technical challenges overcome
- Dense layer optimization to fit into 2 MB Flash  
- Managing inference time (~6s/image)  

---