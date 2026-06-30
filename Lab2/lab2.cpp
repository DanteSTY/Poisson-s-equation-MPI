#include <mpi.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

const double A_PARAM = 1.0e5;
const double EPS = 1.0e-8;
const double X0 = -1.0, Y0 = -1.0, Z0 = -1.0;
const double DX = 2.0, DY = 2.0, DZ = 2.0;

double get_phi_exact(double x, double y, double z) {
    return x * x + y * y + z * z;
}

double get_rho(double x, double y, double z) {
    return 6.0 - A_PARAM * get_phi_exact(x, y, z);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    //size-колво процессов выделенных для работы rank-порядковый номер текущего процесса от 0..N-1,N-колво ядер
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 512;
    if (argc > 1) N = atoi(argv[1]);

    int Nx = N, Ny = N, Nz = N;
    double hx = DX / (Nx - 1), hy = DY / (Ny - 1), hz = DZ / (Nz - 1);//шаги сетки
    double hx2 = hx * hx, hy2 = hy * hy, hz2 = hz * hz;
    double factor = 1.0 / (2.0 / hx2 + 2.0 / hy2 + 2.0 / hz2 + A_PARAM);

    int k_start = (Nz * rank) / size;
    int k_end = (Nz * (rank + 1)) / size;
    int local_Nz = k_end - k_start;//одномерная декомпозиция разрезаем куб по оси z на Nz слоев
    int layer_size = Nx * Ny;

    std::vector<double> phi((size_t)(local_Nz + 2) * layer_size, 0.0);
    std::vector<double> phi_new((size_t)(local_Nz + 2) * layer_size, 0.0);

    // Инициализация гр.условий
    for (int k_loc = 0; k_loc < local_Nz + 2; ++k_loc) {//k_loc локальный индекс слоя  внутри конкретного процесса.
        int k_glob = k_start + k_loc - 1;//k_glob это глобальный индекс слоя во всей расчетной сетке
        double z = Z0 + k_glob * hz;//переводит индекс точки (i, j, k) в реальные координаты в пространстве (x, y, z). 
        for (int j = 0; j < Ny; ++j) {
            double y = Y0 + j * hy;
            for (int i = 0; i < Nx; ++i) {
                double x = X0 + i * hx;
                size_t idx = (size_t)k_loc * layer_size + j * Nx + i;
                if (k_glob <= 0 || k_glob >= Nz - 1 || j == 0 || j == Ny - 1 || i == 0 || i == Nx - 1) {
                    phi[idx] = phi_new[idx] = get_phi_exact(x, y, z);
                }
            }
        }
    }

    // Переменные для профилирования
    double t_start_loop = MPI_Wtime();
    double total_internal_compute = 0;
    double total_wait_time = 0;
    int iterations = 0;
    double global_max_diff = 0;

    do {
        iterations++;
        double max_diff = 0;
        MPI_Request reqs[4];
        int req_count = 0;

        // (I-immediate-немедленный возврат чтобы не ждать пока сосед подтвердит прием(неблокирующие операции)
        if (rank > 0) {//нулевой процесс пропускает это все остальные нет т.к тут обмен слоями с соседом сверху ждем данные от соседа сверху и когда они придут кладем в первый слой(индекс 0)-верхний теневой слой 
            MPI_Irecv(&phi[0], layer_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Isend(&phi[layer_size], layer_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        if (rank < size - 1) {//последний процесс пропускает это обмен с соседом снизу 
            MPI_Irecv(&phi[(size_t)(local_Nz + 1) * layer_size], layer_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Isend(&phi[(size_t)local_Nz * layer_size], layer_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        }

        // перекрытие,res-формула итер.процесса метода якоби
        double t1 = MPI_Wtime();// Засекаем время сразу после запуска обменов
        for (int k_loc = 2; k_loc < local_Nz; ++k_loc) {//пропускаем первый и посл.слои т.к там нужны данные от соседей которые летят по сети ч/з MPI_Isend/Irecv
            int k_glob = k_start + k_loc - 1;
            double z = Z0 + k_glob * hz;
            for (int j = 1; j < Ny - 1; ++j) {
                double y = Y0 + j * hy;
                size_t off = (size_t)k_loc * layer_size + j * Nx;
                for (int i = 1; i < Nx - 1; ++i) {
                    double x = X0 + i * hx;
                    size_t idx = off + i;
                    double res = ((phi[idx - 1] + phi[idx + 1]) / hx2 + (phi[idx - Nx] + phi[idx + Nx]) / hy2 + (phi[idx - layer_size] + phi[idx + layer_size]) / hz2 - get_rho(x, y, z)) * factor;//factor-коэф.полученный из аппрокс 2 произв
                    phi_new[idx] = res;
                    max_diff = std::max(max_diff, std::abs(res - phi[idx]));//max изм-е значения в сетке на текущей итерации
                }
            }
        }
        total_internal_compute += (MPI_Wtime() - t1);

        
        double t2 = MPI_Wtime();// Фиксируем время окончания расчёта внутр.слоев
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);//ф-я синхронизатор когда нужно посчитать граничные слои не считать их пока все данные не долетели
        total_wait_time += (MPI_Wtime() - t2);// Накапливаем чистое время работы процессора

        // Граничные слои
        int bounds[] = { 1, local_Nz };
        for (int k_loc : bounds) {
            int k_glob = k_start + k_loc - 1;
            if (k_glob <= 0 || k_glob >= Nz - 1 || local_Nz < 1) continue;
            double z = Z0 + k_glob * hz;
            for (int j = 1; j < Ny - 1; ++j) {
                double y = Y0 + j * hy;
                size_t off = (size_t)k_loc * layer_size + j * Nx;
                for (int i = 1; i < Nx - 1; ++i) {
                    double x = X0 + i * hx;
                    size_t idx = off + i;
                    double res = ((phi[idx - 1] + phi[idx + 1]) / hx2 + (phi[idx - Nx] + phi[idx + Nx]) / hy2 + (phi[idx - layer_size] + phi[idx + layer_size]) / hz2 - get_rho(x, y, z)) * factor;
                    phi_new[idx] = res;
                    max_diff = std::max(max_diff, std::abs(res - phi[idx]));
                }
            }
        }

        MPI_Allreduce(&max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);//всем процессам говорится какая макс ошибка чтобы процессы знали когда остановится 
        phi.swap(phi_new);

    } while (global_max_diff > EPS);

    double total_time = MPI_Wtime() - t_start_loop;

    // Сбор результатов профилирования на 0 процессе
    double max_wait, avg_compute;
    MPI_Reduce(&total_wait_time, &max_wait, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);//говорит главному процессу какая макс ошибка 
    MPI_Reduce(&total_internal_compute, &avg_compute, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n--- Profiling Results (N=" << N << ", Procs=" << size << ") ---\n";
        std::cout << "Total iterations:  " << iterations << "\n";
        std::cout << "Total loop time:   " << total_time << " s\n";
        std::cout << "Internal Compute:  " << avg_compute / size << " s (avg per proc)\n";
        std::cout << "Wait Time:         " << max_wait << " s (max overhead)\n";

        double overlap_ratio = (1.0 - (max_wait / total_time)) * 100.0;
        std::cout << "Overlap Efficiency: " << std::fixed << std::setprecision(2) << overlap_ratio << "%\n";

        if (max_wait < (avg_compute / size)) {
            std::cout << "Status: SUCCESS. Communication is effectively hidden.\n";
        }
        else {
            std::cout << "Status: WARNING. Communication dominates. Increase N.\n";
        }
    }

    MPI_Finalize();
    return 0;
}