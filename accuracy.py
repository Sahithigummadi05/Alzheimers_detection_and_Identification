get_ipython().magic(u'matplotlib inline')


file_out = open("out.txt","r+")
x_epoch=[]
y=[]
accuracy=[]
acq = []
for line in file_out:
    if 'Epoch' in line:
        m = line.split()
        x_epoch.append(int(m[1]))
        y.append(float(m[7]))
        
    if 'Accuracy' in line:
        acc= line.split()
        accuracy.append(float(acc[1])*100.0)
        acq.append(acc[1])

print("Unhappiness score of the model is {}".format( sum(y)/545 ))


import matplotlib.pyplot as plt

plt.plot(x_epoch, 'bo', y,'k')
#plt.axis([0, 600, 0, 80000])
plt.ylabel("loss function score")
plt.xlabel("Number of Epoch")
plt.show()


import matplotlib.pyplot as plt

plt.plot(x_epoch,accuracy,'.')
#plt.axis([0, 600, 0, 80000])
plt.ylabel("Accuracy Percentage")
plt.xlabel("Number of Epoch")
plt.show()


fig = plt.figure()
fig.set_size_inches(16, 12)
ax = plt.axes()

plt.ylabel("Accuracy Percentage")
plt.xlabel("Number of Epoch")

ax.plot(x_epoch, accuracy)

plt.show()

fig.savefig('accuracy.png')


fig = plt.figure()
fig.set_size_inches(16, 12)
ax = plt.axes()

plt.ylabel("loss function score")
plt.xlabel("Number of Epoch")

plt.ylim(1, 1000)
plt.xlim(1, 600)


ax.plot(x_epoch, y)

fig.savefig('loss.png')