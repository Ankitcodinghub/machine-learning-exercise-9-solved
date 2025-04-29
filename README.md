# machine-learning-exercise-9-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning Exercise 9 Solved](https://www.ankitcodinghub.com/product/machine-learning-labs-solved-6/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;110203&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning  Exercise 9 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
Goals. The goals of this exercise are to:

• Define and train convolutional and residual networks in PyTorch.

• Explore some algorithmic properties of Adam and SGD with momentum.

• Analyze the receptive field of convolutional networks.

Problem 1 (Convolutional and Residual Networks in PyTorch):

The accompaning Jupyter Notebook has two PyTorch coding excercises. We recommend running the notebook on Google Colab which provides you with a free GPU and does not require installing any packages.

1. Open the colab link for the lab 9:

https://colab.research.google.com/github/epfml/ML_course/blob/master/labs/ex09/template/ex09.ipynb 2. To save your progress, click on “File &gt; Save a Copy in Drive” to get your own copy of the notebook.

3. Click ‘connect’ on top right to make the notebook executable (or ‘open in playground’).

4. Work your way through the introduction and exercises.

Alternatively you can download the notebook from GitHub and install PyTorch locally, see the instructions on pytorch.org.

Problem 2 (Adam and SGDM):

SGD with momentum (SGDM) and Adam are two very commonly used optimizers in deep learning. Both are example of first order optimization methods that update the weights based on their gradients after some processing. The two algorithms are given below. Note that both of these algorithms act on each scalar parameter independently, and do not consider whether a parameter is a part of a larger vector/matrix/tensor.

Adam:

(1)

(2)

(3)

(4)

(5)

SGDM:

m(wt+1) ← βm(wt) +∇wL(t) w(t+1) ← w(t) − ηm(wt+1)

(6)

(7)

For both algorithms, L(t) is the loss for time t (typically this is the loss for a mini-batch of samples), and w(t) represents the value of the parameter at step t. The algorithm shows an update for a single parameter but all model parameters are updated in the same way for each timestep t. Both optimizers use an exponential moving average of the gradient called momentum (represented by ). Adam additionally uses an exponential moving average of the square gradient ( ) and also computes a “bias correction” for m and v given by mˆ and vˆ. In both cases we consider the intial state to be zero, i.e. . The hyperparameters and their possible values are η ≥ 0,0 &lt; β1 &lt; 1,0 &lt; β2 &lt; 1,ϵ ≥ 0 for Adam and η ≥ 0,0 &lt; β &lt; 1 for SGDM.

1. How many values does each optimizer need to store for a given parameter to perform the next update? This factor determines the memory usage of the optimizer.

2. Let’s assume the gradient is a constant ∇wL(t) = c &gt; 0 for all t ≥ 0 and ϵ = 0. Compute the value of w,mw,vw,mˆw,vˆw for timestep t and both optimizers (where applicable). Assume w(0) = 0 for this question. How does w depend on c in each case?

Problem 3 (Receptive Field of Convolutions):

Convolutions can occur in one or more dimensions. In class you learned about 2D convolutions but both 1D and 3D convolutions are used in certain areas as well (for signals of a corresponding dimension). 1D convolutions are easy to visualize and many insights about them generalize to higher dimensions. You can view a 1D convolution as a special case of 2D convolution where the height of the input and filter is equal to 1.

In this exercise we will explore how the size of the output depends on the input size and parameters of a convolution in 1D. We will then use this to analyze the receptive field of a convolutional networks. The receptive field of a given activation is the area of the input that can affect its value. This is important to keep in mind when working with convolutional networks since the receptive field must be sufficiently large for certain features to be learned. For example, if your convolutional network was looking for a certain phrase in an audio signal, the receptive field of the later neurons should be sufficiently long to cover the length of the phrase.

The output size of a 1D convolution is depends on the dimensions of the input as well as the kernel size K, the padding P and the stride S. Padding is applied to both sides of the input signal and adds P values to each side, typically zeros (but various other forms of padding also exist). After adding a given amount of padding, a convolution only computes outputs where the filter can be fully “overlayed” on the padded input signal. A convolution with stride S only computes every S-th element of the output (starting with the first valid position on the edge). In modern networks we often use strided convolutions instead of adding pooling layers.

• Let’s assume we have a 1D convolution with input X of width Win, a kernel size of K, padding P and stride S. What is the size Wout of the output Y ?

• Given an output size Wout for the convolution above, what is the minimum size of the input, Win?

• Given a sequence of L convolutions with kernel sizes K(1),…K(L), padding P(1),…P(L), and strides S(1),…S(L), what is the receptive field of an output element of the last convolution? You can assume that the input is larger than the receptive field (otherwise the definition is unclear).

Hint: Does padding affect the receptive field? Start with an output width of 1 and work your way backwards using the results of the previous parts. You don’t have to simplify the resulting recurrence relation.

2
