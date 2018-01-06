import theano
from theano import tensor as T
import numpy as np

x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = w1 * x1 + w0

net_input = theano.function(inputs=[w1, x1, w0], outputs=z1)

print('Net input: %.2f' % net_input(2.0, 1.0, 0.5))

print(theano.config.floatX)
print(theano.config.device)

# Initialize
x = T.dmatrix(name='x')
x_sum = T.sum(x, axis=0)

# Compile
calc_sum = theano.function(inputs=[x], outputs=x_sum)

# Execute
ary = [[1, 2, 3], [1, 2, 3]]
print('Column sum:', calc_sum(ary))

ary = np.array([[1, 2, 3], [1, 2, 3]], dtype=theano.config.floatX)
print('Column sum:', calc_sum(ary))

######################

# Initialize

x = T.dmatrix('x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], dtype=theano.config.floatX))

z = x.dot(w.T)
update = [[w, w + 1.0]]

# Compile
net_input = theano.function(inputs=[x], updates=update, outputs=z)

# Execute
data = np.array([[1, 2, 3]], dtype=theano.config.floatX)

for i in range(5):
    print('z%d:' % i, net_input(data))
