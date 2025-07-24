https://peterroelants.github.io/posts/gaussian-process-tutorial/

A gaussian process models a stochastic (random) process, where the variables follow the gaussian distribution. Each 'realisation' of a gaussian process is a function. In practise, we sample the function at regular points to see the full shape. Each realisation(function) is evaluated at 'n' points, and will be a n-dimensional multivariate gaussian. It has a mean function, and a kernel function. These can define the means and covariances for arbitrary number of points. If we have n observations, they will help define a n-dim multivariate normal. When evaluating on m points, they will have a m-dimensional multivariate normal distribution. this is called 'marginalising' from the GP, which is evaluated at a fixed number of points, even though it can be done for infinite points in principle. Given the observation points, you can calculate the fn values at the unobserved points along with associated uncertainty, by calculating the funtion given the observed x,y, and testx.

Noise can be incorporated by adding the noise term to the diagonal of the matrix.

Gps in tinygp:

- diag term in gp: the per observation observation error
- key operations: conditioning and marginalisation
  - conditioning: train on observed data points, condition to get the log prob as well as another gp that defines the behaviour at test points

> For example, you may want to fit for the parameters of your kernel model (the length scale and amplitude, for example), and a good objective to use for that process is the marginal likelihood of the process evaluated for the observed data.
