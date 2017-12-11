import numpy as np

def adapt_thresh(df,pre=8,post=7):
    df = df.T
    alpha = 9
    thresh = 0.033
    fn = np.mean

    N = len(df)
    m = np.zeros(min(post,N))
    for i in range(0,len(m)):
        k = min(i + pre, N)
        m[i] = fn(df[0:k])

    if N > (post + pre):
        m = np.hstack((m, fn(_buffer(df.T, post + pre + 1, post + pre, 'nodelay'), axis=0)[pre + post + 1:]))

    for i in N-1+np.arange(-pre,0):
        j = max(i - post, 1)
        m = np.hstack((m,fn(df[j:]).T))

    df = df - m
    dfout = (df > 0)* df

    return dfout





def _buffer(x, n, p=0, opt=None):
    '''Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

    Args
    ----
    x:   signal array
    n:   number of data segments
    p:   number of values to overlap
    opt: initial condition options. default sets the first `p` values
         to zero, while 'nodelay' begins filling the buffer immediately.
    '''
    import numpy

    if p >= n:
        raise ValueError('p ({}) must be less than n ({}).'.format(p,n))

    # Calculate number of columns of buffer array
    cols = int(numpy.ceil(len(x)/(n-p)))

    # Check for opt parameters
    if opt == 'nodelay':
        # Need extra column to handle additional values left
        cols += 1
    elif opt != None:
        raise SystemError('Only `None` (default initial condition) and '
                          '`nodelay` (skip initial condition) have been '
                          'implemented')

    # Create empty buffer array
    b = numpy.zeros((n, cols))

    # Fill buffer by column handling for initial condition and overlap
    j = 0
    for i in range(cols):
        # Set first column to n values from x, move to next iteration
        if i == 0 and opt == 'nodelay':
            b[0:n,i] = x[0:n]
            continue
        # set first values of row to last p values
        elif i != 0 and p != 0:
            b[:p, i] = b[-p:, i-1]
        # If initial condition, set p elements in buffer array to zero
        else:
            b[:p, i] = 0

        # Get stop index positions for x
        k = j + n - p

        # Get stop index position for b, matching number sliced from x
        n_end = p+len(x[j:k])

        # Assign values to buffer array from x
        b[p:n_end,i] = x[j:k]

        # Update start index location for next iteration of x
        j = k

    return b