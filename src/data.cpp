#include <data.h>
#include <common.h>

Data::Data(py::array_t<double> x, py::array_t<double> y, int num_threads)
{
    omp_set_num_threads(num_threads);
    // copy the x
    auto x_unchecked = x.unchecked<2>();
    size_t N = x_unchecked.shape(0), D = x_unchecked.shape(1);
    this->N = N;
    this->D = D;
    this->x = std::vector<double>(N*D);
    // go through the array, column by column, filling things up in fortran order
    // (column-major)
    #pragma omp parallel for schedule(static) collapse(2)
    for(size_t j=0; j<D; j++)
    {
        for(size_t i=0; i<N; i++)
        {
            this->x[j*N+i] = x_unchecked(i,j);
        }
    }
    // standardize the data
    compute_mean_std_and_standardize_x_data();
    // copy the y
    auto y_unchecked = y.unchecked<1>();
    this->y = std::vector<double>(N);
    // go through the array, column by column, filling things up in fortran order
    // (column-major)
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<N; i++)
    {
        this->y[i] = y_unchecked(i);
    }
};

void Data::compute_mean_std_and_standardize_x_data()
{
    // initialize this->means and this->stds
    means = std::vector<double>(D);
    stds = std::vector<double>(D);

    // compute the means and stds for each dimension
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<D; j++)
    {
        double mean=0, meansq=0;
        int n = 1;
        for(size_t i=j*N; i<(j+1)*N; i++)
        {
            mean += (x[i]-mean)/n;
            meansq += (x[i]*x[i]-meansq)/n;
            n++;
        }
        // store them in the relevant arrays
        means[j] = mean;
        stds[j] = std::sqrt(meansq-mean*mean);
    }
    // now normalize each column by subtracting the mean and dividing by the std
    #pragma omp parallel for schedule(static)
    for(size_t j=0; j<D; j++)
    {
        for(size_t i=j*N; i<(j+1)*(N); i++)
        {
            x[i] = (x[i]-means[j])/stds[j];
        }
    }
};