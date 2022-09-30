#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <iomanip>
#include <pthread.h>
#include <chrono>
using namespace std;

const int    NUMBER_OF_THREADS = 10;
const string FEATURE           = "GrLivArea",
             TARGET            = "SalePrice",
             FILENAME_PREFIX   = "dataset_",
             FILENAME_SUFFIX   = ".csv";

struct thread_arguments
{
    int feature_index;
    int target_index;
    int predict_index;
    int threshold;
    string full_directory;
    vector < vector <int> > rows;
    vector <float> u0;
    vector <float> s0;
    vector <float> d0;
    vector <float> u1;
    vector <float> s1;
    vector <float> d1;
    vector <float> std0;
    vector <float> std1;
    float accuracy;
    int n0;
    int n1;
};


vector < vector <int> > read_csv(string full_directory, int *feature_index, int *target_index, int *predict_index)
{
    ifstream ifs;
    ifs.open(full_directory, ifstream::in);
    string line, word;
    vector <int> row;
    vector < vector <int> > rows;
    bool first_row = true;
    while (getline(ifs, line))
    {
        stringstream ss (line);
        if (first_row)
        {
            int i = 0;
            bool found_target = false;
            while (getline(ss, word, ','))
            {
                if (word == FEATURE)
                   *feature_index = i;
                else if (word == TARGET)
                {
                    *target_index = i;
                    found_target = true;
                }
                i ++;
            }
            if (found_target == false)
                *target_index = i - 1;
            *predict_index = i;
            first_row = false;
            continue;
        }
        row.clear();
        while (getline(ss, word, ','))
            row.push_back(stoi(word));
        row.push_back(0);
        rows.push_back(row);
    }
    ifs.close();
    return rows;
}

vector < vector <int> > label(const vector < vector <int> > &rows, int target_index, int threshold, int *n0, int *n1)
{
    vector < vector <int> > tmp = rows;
    int len = tmp.size(), _n0 = 0, _n1 = 0;
    for (int i = 0; i < len; i ++)
    {
        if (tmp[i][target_index] < threshold)
        {
            tmp[i][target_index] = 0;
            _n0 ++;
        }
        else
        {
            tmp[i][target_index] = 1;
            _n1 ++;
        }
    }
    *n0 = _n0;
    *n1 = _n1;
    return tmp;
}

vector <float> csv_mean(const vector < vector <int> > &rows, vector <float> &sum, int target_index, int number_of_fields, int label)
{
    int n = 0, len = rows.size();
    vector <float> s (number_of_fields, 0);
    for (int i = 0; i < len; i ++)
    {
        if (rows[i][target_index] == label)
        {
            for (int field = 0; field < number_of_fields; field ++)
                s[field] += rows[i][field];
            n ++;
        }
    }
    for (int field = 0; field < number_of_fields; field ++)
    {
        sum[field] = s[field];
        s[field] /= n;
    }
    return s;
}

vector <float> csv_std(const vector < vector <int> > &rows, const vector <float> &u, vector <float> &d, int target_index, int number_of_fields, int label)
{
    int n = 0, len = rows.size();
    vector <float> s (number_of_fields, 0);
    for (int i = 0; i < len; i ++)
    {
        if (rows[i][target_index] == label)
        {
            for (int field = 0; field < number_of_fields; field ++)
                s[field] += pow(rows[i][field] - u[field], 2);
            n ++;
        }
    }
    for (int field = 0; field < number_of_fields; field ++)
    {
        d[field] = s[field];
        s[field] = sqrt(s[field] / n);
    }
    return s;
}

vector < vector <int> > predict(const vector < vector <int> > &rows, int feature_index, int predict_index, float u, float std)
{
    vector < vector <int> > tmp = rows;
    int len = tmp.size();
    float lowest_value = u - std, highest_value = u + std;
    for (int i = 0; i < len; i ++)
    {
        int feature_value = tmp[i][feature_index];
        if (feature_value < highest_value && feature_value > lowest_value)
            tmp[i][predict_index] = 1;
    }
    return tmp;
}

float calculate_accuracy(const vector < vector <int> > &rows, int target_index, int predict_index)
{
    int len = rows.size();
    float number_of_correctly_classified_samples = 0;
    for (int i = 0; i < len; i ++)
        if (rows[i][target_index] == rows[i][predict_index])
            number_of_correctly_classified_samples ++;
    return 100 * number_of_correctly_classified_samples / len;
}

void *csv_thread_handler(void *arg)
{
    thread_arguments *args = (thread_arguments *) arg;
    int feature_index, target_index, predict_index;
    string full_directory = args->full_directory;
    int threshold = args->threshold;
    vector < vector <int> > rows = read_csv(full_directory, &feature_index, &target_index, &predict_index);
    args->feature_index = feature_index;
    args->target_index = target_index;
    args->predict_index = predict_index;
    int n0, n1;
    rows = label(rows, target_index, threshold, &n0, &n1);
    args->n0 = n0;
    args->n1 = n1;
    args->rows = rows;
    int number_of_fields = predict_index;
    vector <float> s0 (number_of_fields, 0), s1 (number_of_fields, 0);
    vector <float> u0 = csv_mean(rows, s0, target_index, number_of_fields, 0);
    vector <float> u1 = csv_mean(rows, s1, target_index, number_of_fields, 1);
    args->u0 = u0;
    args->s0 = s0;
    args->u1 = u1;
    args->s1 = s1;
    pthread_exit(NULL);
}

void *std_thread_handler(void *arg)
{
    thread_arguments *args = (thread_arguments *) arg;
    vector <float> u0 = args->u0;
    vector <float> u1 = args->u1;
    vector < vector <int> > rows = args->rows;
    int target_index = args->target_index, number_of_fields = args->predict_index;
    vector <float> d0 (number_of_fields, 0), d1 (number_of_fields, 0);
    vector <float> std0 = csv_std(rows, u0, d0, target_index, number_of_fields, 0);
    vector <float> std1 = csv_std(rows, u1, d1, target_index, number_of_fields, 1);
    args->std0 = std0;
    args->d0 = d0;
    args->std1 = std1;
    args->d1 = d1;
    pthread_exit(NULL);
}

void *predict_thread_handler(void *arg)
{
    thread_arguments *args = (thread_arguments *) arg;
    vector < vector <int> > rows = args->rows;
    int feature_index = args->feature_index, target_index = args->target_index, predict_index = args->predict_index;
    float u = args->u1[feature_index], std = args->std1[feature_index];
    rows = predict(rows, feature_index, predict_index, u, std);
    args->rows = rows;
    args->accuracy = calculate_accuracy(rows, target_index, predict_index);
    pthread_exit(NULL);
}

vector <float> mean_handler(thread_arguments *arg[], int label)
{
    int number_of_fields = arg[0]->predict_index;
    vector <float> s (number_of_fields, 0);
    int N = 0;
    for (int tid = 0; tid < NUMBER_OF_THREADS; tid ++)
    {
        vector <float> sum = label ? arg[tid]->s1 : arg[tid]->s0;
        int n = label ? arg[tid]->n1 : arg[tid]->n0;
        for (int field = 0; field < number_of_fields; field ++)
            s[field] += sum[field];
        N += n;
    }
    for (int field = 0; field < number_of_fields; field ++)
        s[field] /= N;
    return s;
}

vector <float> std_handler(thread_arguments *arg[], int label)
{
    int number_of_fields = arg[0]->predict_index;
    vector <float> s (number_of_fields, 0);
    int N = 0;
    for (int tid = 0; tid < NUMBER_OF_THREADS; tid ++)
    {
        vector <float> d = label ? arg[tid]->d1 : arg[tid]->d0;
        int n = label ? arg[tid]->n1 : arg[tid]->n0;
        for (int field = 0; field < number_of_fields; field ++)
            s[field] += d[field];
        N += n;
    }
    for (int field = 0; field < number_of_fields; field ++)
        s[field] = sqrt(s[field] / N);
    return s;
}

float accuracy_handler(thread_arguments *arg[])
{
    float s = 0;
    int N = 0;
    for (int tid = 0; tid < NUMBER_OF_THREADS; tid ++)
    {
        float accuracy = arg[tid]->accuracy;
        int n = (arg[tid]->rows).size();
        s += n * accuracy;
        N += n;
    }
    return s / N;
}

int main(int argc, char *argv[])
{
    // chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (argc < 3)
    {
        cerr << "Not enough input arguments." << endl;
        return 1;
    }
    string directory = argv[1];
    int threshold = stoi(argv[2]);
    thread_arguments *arg[NUMBER_OF_THREADS];
    pthread_t tid[NUMBER_OF_THREADS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    for (int i = 0; i < NUMBER_OF_THREADS; i ++)
    {
        arg[i] = new thread_arguments;
        arg[i]->full_directory = directory + FILENAME_PREFIX + to_string(i) + FILENAME_SUFFIX;
        arg[i]->threshold = threshold;
        if (pthread_create(&tid[i], &attr, csv_thread_handler, (void *) arg[i]))
        {
            cerr << "Something bad happened. That's all we know." << endl;
            exit(1);
        }
    }
    void *status;
    for (int i = 0; i < NUMBER_OF_THREADS; i ++)
    {
        if (pthread_join(tid[i], &status))
        {
            cerr << "Something bad happened. That's all we know." << endl;
            exit(1);
        }
    }
    vector <float> u0 = mean_handler(arg, 0);
    vector <float> u1 = mean_handler(arg, 1);
    for (int i = 0; i < NUMBER_OF_THREADS; i ++)
    {
        arg[i]->u0 = u0;
        arg[i]->u1 = u1;
        if (pthread_create(&tid[i], &attr, std_thread_handler, (void *) arg[i]))
        {
            cerr << "Something bad happened. That's all we know." << endl;
            exit(1);
        }
    }
    for (int i = 0; i < NUMBER_OF_THREADS; i ++)
    {
        if (pthread_join(tid[i], &status))
        {
            cerr << "Something bad happened. That's all we know." << endl;
            exit(1);
        }
    }
    vector <float> std0 = std_handler(arg, 0);
    vector <float> std1 = std_handler(arg, 1);
    for (int i = 0; i < NUMBER_OF_THREADS; i ++)
    {
        arg[i]->std0 = std0;
        arg[i]->std1 = std1;
        if (pthread_create(&tid[i], &attr, predict_thread_handler, (void *) arg[i]))
        {
            cerr << "Something bad happened. That's all we know." << endl;
            exit(1);
        }
    }
    for (int i = 0; i < NUMBER_OF_THREADS; i ++)
    {
        if (pthread_join(tid[i], &status))
        {
            cerr << "Something bad happened. That's all we know." << endl;
            exit(1);
        }
    }
    float accuracy = accuracy_handler(arg);
    cout << "Accuracy: " << fixed << setprecision(2) << accuracy << "%";
    // chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // cout << "\nTime difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    pthread_exit(NULL);
}