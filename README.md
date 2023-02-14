[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/hi_tysam.svg?style=social&label=Follow%20%40TySam_And)](https://twitter.com/hi_tysam)

## CIFAR10 hyperlightspeedbench
Welcome to the hyperlightspeedbench CIFAR-10 (HLB-CIFAR10) repo.

### How to Run


`git clone https://github.com/tysam-code/hlb-CIFAR10 && cd hlb-CIFAR10 && python -m pip install -r requirements.txt && python main.py`


If you're curious, this code is generally Colab friendly (in fact -- most of this was developed in Colab!). Just be sure to uncomment the reset block at the top of the code.


### Main

Goals:

* minimalistic
* beginner-friendly
* torch- and python-idiomatic
* hackable
* few external dependencies (currently only torch and torchvision)
* ~world-record single-GPU training time (this repo holds the current world record at ~<10 seconds on an A100, down from ~18.1 seconds originally).
* <2 seconds training time in <2 years (yep!)

This is a neural network implementation of a very speedily-training network that originally started as a painstaking reproduction of [David Page's original ultra-fast CIFAR-10 implementation on a single GPU](https://myrtle.ai/learn/how-to-train-your-resnet/), but written nearly from the ground-up to be extremely rapid-experimentation-friendly. Part of the benefit of this is that we now hold the world record for single GPU training speeds on CIFAR10 (under 10 seconds on an A100!!!)

What we've added:
* squeeze and excite layers
* way too much hyperparameter tuning
* miscellaneous architecture trimmings (see the patch notes)
* memory format changes (and more!) to better use tensor cores/etc
* and more!

This code, in comparison to David's original code, is in a single file and extremely flat, but is not as durable for long-term production-level bug maintenance. You're meant to check out a fresh repo whenever you have a new idea. It is excellent for rapid idea exploring -- almost everywhere in the pipeline is exposed and built to be user-friendly. I truly enjoy personally using this code, and hope you do as well! :D Please let me know if you have any feedback. I hope to continue publishing updates to this in the future, so your support is encouraged. Share this repo with someone you know that might like it!

Feel free to check out my[Patreon](https://www.patreon.com/user/posts?u=83632131) if you like what I'm doing here and want more!. Additionally, if you want me to work up to a part-time amount of hours with you, feel free to reach out to me at hire.tysam@gmail.com. I'd love to hear from you.


### Known Bugs

The Colab-specific code is commented out at the top, and the timing/performance table reprints the entire table instead of appropriately updating in-place each epoch.

### Why a ConvNet Still? Why CIFAR10? Aren't Transformers the New Thing Now?


Transformers are indeed the new thing, but I personally believe that the way information condenses from a training set into a neural network will still practically always follow the same underlying set of mathematical principles. The goal for this codebase is to get training in under two (2) seconds within a year or two (2), and under one (1) seconds within 4-5 years. This should allow for some very interesting scaled experiments for different techniques on a different kind of level. I have a rough path planned down to about 2-3 seconds of training or so, all things working out as they should. It will likely get very, very difficult beyond that point.

Basically -- the information gained from experimenting with a technique here should translate in some kind of a way. No need to scale up size arbitrarily when looking to suss out the basics of certain applied mathematical concepts for a problem.


### Submissions

Currently, submissions to this codebase as a benchmark are closed as we figure out the level of interest, how to track disparate entries, etc. Feel free to open an issue if you have thoughts on this!

#### Bugs & Etc.

If you find a bug, open an issue! L:D If you have a success story, let me know! It helps me understand what works and doesn't more than you might expect -- if I know how this is specifically helping people, that can help me further improve as a developer, as I can keep that in mind when developing other software for people in the future. :D :)

### Baselines

#### RTX 3090

Mean, StdDev, Min, Max Accuracy: (0.9400334358215332, 0.0010016737505793571, 0.9389001131057739, 0.940800130367279)
Mean, StdDev, Min, Max Train Time: (22.152647018432617, 0.5368512868881226, 21.831771850585938, 22.772418975830078)
Eval time/epoch mean(s): 0.23741876220703123
Wall clock(s): 79.8479734

#### A100

Mean, StdDev, Min, Max Accuuracy: (0.9397000670433044, 0.0012124303029850125, 0.9390000700950623, 0.9411000609397888)
Mean, StdDev, Min, Max Train Time: (9.701638221740723, 0.047454722225666046, 9.651701927185059, 9.74614429473877)
Eval time/epoch mean(s): 0.10091776008605956
Wall clock(s): 34.097338891006075


##### NVIDIA GeForce RTX 3080 Laptop GPU
Memory Usage:
Memory Allocated: 0.5 GB
Memory Cached:    7.3 GB
Mean, StdDev, Min, Max Accuracy: (0.9424667358398438, 0.0022479381877928972, 0.940000057220459, 0.9444000124931335)
Mean, StdDev, Min, Max Train Time: (44.40179443359375, 0.17301391065120697, 44.29471969604492, 44.60139465332031)
Eval time/epoch mean(s): 0.5512141811370849
Wall clock(s): 155.6284537