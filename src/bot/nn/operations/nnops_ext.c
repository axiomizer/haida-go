#include <Python.h>
#include <numpy/arrayobject.h>

static int parse_arg(PyObject *object, PyArrayObject **array) {
    PyArray_Descr *dtype = PyArray_DescrFromType(NPY_DOUBLE);
    *array = (PyArrayObject *)PyArray_FromAny(object, dtype, 2, 2, NPY_ARRAY_IN_ARRAY, NULL);
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

    for (int i = 0; i < out_dims[0]; i++) {
        for (int j = 0; j < out_dims[1]; j++) {
            double sum = 0;
            int i2_base1 = overhang > i ? overhang - i : 0;
            int i2_base2 = overhang > j ? overhang - j : 0;
            int i1_base1 = i > overhang ? i - overhang : 0;
            int i1_base2 = j > overhang ? j - overhang : 0;
            int temp = (int)in1_dims[0] + overhang - i;
            int len1 = (temp < (int)in2_dims[0] ? temp : (int)in2_dims[0]) - i2_base1;
            temp = (int)in1_dims[1] + overhang - j;
            int len2 = (temp < (int)in2_dims[1] ? temp : (int)in2_dims[1]) - i2_base2;
            for (int m = 0; m < len1; m++) {
                for (int n = 0; n < len2; n++) {
                    npy_double item1 = i2_data[(i2_base1 + m) * in2_dims[1] + i2_base2 + n];
                    npy_double item2 = i1_data[(i1_base1 + m) * in1_dims[1] + i1_base2 + n];
                    sum += item1 * item2;
                }
            }
            o_data[i * out_dims[1] + j] = sum;
        }
    }

    Py_DECREF(input1);
    Py_DECREF(input2);
    return (PyObject *)output;
}

static PyMethodDef methods[] = {
    {"correlate", method_correlate, METH_VARARGS, "Calculate correlation between two numpy arrays"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "nnops_ext",
    "Fast matrix operations for a neural network",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_nnops_ext(void) {
    import_array();
    return PyModule_Create(&module_def);
}
