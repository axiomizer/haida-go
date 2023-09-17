#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdbool.h>

static void correlate(npy_double *i1_data, int i1d0, int i1d1,
                      npy_double *i2_data, int i2d0, int i2d1,
                      npy_double *o_data,  int od0,  int od1,
                      int overhang, bool pluseq) {
    for (int i = 0; i < od0; i++) {
        for (int j = 0; j < od1; j++) {
            double sum = 0;
            int i2_base1 = overhang > i ? overhang - i : 0;
            int i2_base2 = overhang > j ? overhang - j : 0;
            int i1_base1 = i > overhang ? i - overhang : 0;
            int i1_base2 = j > overhang ? j - overhang : 0;
            int temp = i1d0 + overhang - i;
            int len1 = (temp < i2d0 ? temp : i2d0) - i2_base1;
            temp = i1d1 + overhang - j;
            int len2 = (temp < i2d1 ? temp : i2d1) - i2_base2;
            for (int m = 0; m < len1; m++) {
                for (int n = 0; n < len2; n++) {
                    npy_double item1 = i2_data[(i2_base1 + m) * i2d1 + i2_base2 + n];
                    npy_double item2 = i1_data[(i1_base1 + m) * i1d1 + i1_base2 + n];
                    sum += item1 * item2;
                }
            }
            if (pluseq) {
                o_data[i * od1 + j] += sum;
            } else {
                o_data[i * od1 + j] = sum;
            }
        }
    }
}

static int parse_arg(PyObject *object, PyArrayObject **array) {
    PyArray_Descr *dtype = PyArray_DescrFromType(NPY_DOUBLE);
    *array = (PyArrayObject *)PyArray_FromAny(object, dtype, 0, 0, NPY_ARRAY_IN_ARRAY, NULL);
    return *array != NULL;
}

static PyObject *method_correlate(PyObject *self, PyObject *args) {
    PyArrayObject *input1 = NULL, *input2 = NULL, *output = NULL;
    int overhang;
    if (!PyArg_ParseTuple(args, "O&O&i", parse_arg, &input1, parse_arg, &input2, &overhang)) {
        PyErr_SetString(PyExc_ValueError, "Failed to parse args");
        return NULL;
    }

    npy_intp *in1_dims = PyArray_DIMS(input1);
    npy_intp *in2_dims = PyArray_DIMS(input2);
    npy_intp const out_dims[] = {in1_dims[0] - in2_dims[0] + 1 + (overhang * 2),
                                 in1_dims[1] - in2_dims[1] + 1 + (overhang * 2)};
    output = (PyArrayObject*)PyArray_SimpleNew(2, out_dims, NPY_DOUBLE);
    npy_double *o_data = (npy_double*)PyArray_DATA(output);
    npy_double *i1_data = (npy_double*)PyArray_DATA(input1);
    npy_double *i2_data = (npy_double*)PyArray_DATA(input2);

    correlate(i1_data, (int)in1_dims[0], (int)in1_dims[1],
              i2_data, (int)in2_dims[0], (int)in2_dims[1],
              o_data,  (int)out_dims[0], (int)out_dims[1],
              overhang, false);

    Py_DECREF(input1);
    Py_DECREF(input2);
    return (PyObject *)output;
}

static PyObject *method_correlate_all(PyObject *self, PyObject *args) {
    PyArrayObject *activations = NULL, *kernels = NULL, *output = NULL;
    if (!PyArg_ParseTuple(args, "O&O&", parse_arg, &activations, parse_arg, &kernels)) {
        PyErr_SetString(PyExc_ValueError, "Failed to parse args");
        return NULL;
    }

    npy_intp *activation_dims = PyArray_DIMS(activations);
    int fa1 = activation_dims[2] * activation_dims[3];
    int fa2 = activation_dims[1] * fa1;
    npy_intp *kernel_dims = PyArray_DIMS(kernels);
    int fk1 = kernel_dims[2] * kernel_dims[3];
    int fk2 = kernel_dims[1] * fk1;
    npy_intp const out_dims[] = {activation_dims[0], kernel_dims[1], activation_dims[2], activation_dims[3]};
    output = (PyArrayObject*)PyArray_SimpleNew(4, out_dims, NPY_DOUBLE);
    PyArray_FILLWBYTE(output, 0);
    int fo1 = out_dims[2] * out_dims[3];
    int fo2 = out_dims[1] * fo1;
    npy_double *o_data = (npy_double*)PyArray_DATA(output);
    npy_double *a_data = (npy_double*)PyArray_DATA(activations);
    npy_double *k_data = (npy_double*)PyArray_DATA(kernels);

    for (int ex = 0; ex < activation_dims[0]; ex++) {
        for (int out_f = 0; out_f < kernel_dims[1]; out_f++) {
            for (int in_f = 0; in_f < kernel_dims[0]; in_f++) {
                correlate(&a_data[ex * fa2 + in_f * fa1], activation_dims[2], activation_dims[3],
                          &k_data[in_f * fk2 + out_f * fk1], kernel_dims[2], kernel_dims[3],
                          &o_data[ex * fo2 + out_f * fo1], out_dims[2], out_dims[3],
                          1, true);
            }
        }
    }

    Py_DECREF(activations);
    Py_DECREF(kernels);
    return (PyObject *)output;
}

static PyMethodDef methods[] = {
    {"correlate", method_correlate, METH_VARARGS, "Calculate correlation between two numpy arrays"},
    {"correlate_all", method_correlate_all, METH_VARARGS, "Correlate input activations with kernels"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "nn_ext",
    "Fast matrix operations for a neural network",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_nn_ext(void) {
    import_array();
    return PyModule_Create(&module_def);
}
