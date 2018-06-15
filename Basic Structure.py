import tensorflow as tf

W = tf.Variable([.2],tf.float32)
b = tf.Variable([-.2],tf.float32)

x = tf.placeholder(tf.float32)
linear_model = W*x+b
y = tf.placeholder(tf.float32)

#Defining Loss
squared_Delta = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_Delta)
#Optimizing
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#This is epochs
for i in range (1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})


#print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
print(sess.run([W,b]))