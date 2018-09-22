# Music generation @ hamr2018

Contrapuntal music can be seen as a dialogue between two or more voices.
In the simple case of two-voice counterpoint (but also in countless other musical expressions, e.g. Afro-American music, West-African music, etc.), one can sometimes assimilate the musical interaction of the parts to the so-called 'antiphony' (sometimes referred to as 'call and reponse'): one voice asks a question and a second voice answers.

We would like to come up with a model that is able to understand such a relation from some training data and, when given a musical question as an input, to generate an appropriate answer.

And we want to keep it as simple as possible.

## The idea
Since we see the music as question and answer, we take an approach that works also in language processing.
We feed the entire question as input, note by note, and we generate the entire answer as output, regardless of the fact that in music the answer is typically played at the same time as the question.
This approximation is not inappropriate since a composer, when he or she writes the music, has knowledge of what is going to happen after.

This kind of problem is best solved with recurrent neural networks (RNN) and particularly a sequence to sequence approach.
See for example the following picture, taken from `https://github.com/farizrahman4u/seq2seq`:

![picture](./seq2seq.png)

Here, seq2seq is used with plain English text.
In our case we will substitute words with musical notes but this doesn't change the structure of the architecture nor the training of the network in any way.

The model can be expanded through the use of Neural Turing Machine, which is a form of a memory network (network, that has access to RAM). Neural Turing Machines function much the same way as ordinary
Turing Machines, where a controller decides what to read from the memory and where to move its head. The difference is that a controller is a differentiable function (a neural network). We use an LSTM
with the same architecture as the sequence-to-sequence memoryless model as a controller for our NTM. 

## The training data
We take Bach's two-part inventions as a database for several reasons: 
first of all, it makes a homogeneous body of work who offers some variance while keeping well-defined internal laws;
then, the two-part inventions are very well-known and regarded as a textbook example of contrapuntal music;
they are very simple and could provide an excellent training set for a small model;
and finally, they were already available to us in MIDI format.

Here is an excerpt of a couple of measures from the first invention.

![picture](./data_example.png)

With the idea of making it as simple as possible, we transposed all the songs to the key of C major or A minor.
In this way the network doesn't have to learn complex tonal structures that would most probably need much more data.

## The results
TODO
