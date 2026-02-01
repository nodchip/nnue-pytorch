#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>
#include <mutex>
#include <atomic>
#include <thread>
#include <deque>
#include <random>

#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"

#include "YaneuraOu/source/usi.h"
#include "tanuki_progress.h"

#if defined (__x86_64__)
#define EXPORT
#define CDECL
#else
#if defined (_MSC_VER)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL __attribute__ ((__cdecl__))
#endif
#endif

namespace {
    std::once_flag INITIALIZED;
    Tanuki::Progress TANUKI_PROGRESS;
    std::atomic_bool TANUKI_PROGRESS_LOADED = false;
}

struct HalfKA_hm {
    static constexpr int NUM_SQ = 81;
    static constexpr int INPUTS = 5 * static_cast<int>(FILE_NB) * static_cast<int>(Eval::BonaPiece::e_king);

    static constexpr int MAX_ACTIVE_FEATURES = PIECE_NUMBER_NB;

    static int make_index(Square sq_k, Eval::BonaPiece p) {
        if (sq_k >= SQ_61) {
            // Mirror king square in the enemy camp for symmetry.
            sq_k = Mir(sq_k);

            if (p >= Eval::BonaPiece::fe_hand_end) {
                // Mirror board squares for on-board pieces as well.
                int piece_index = (p - Eval::BonaPiece::fe_hand_end) / SQ_NB;
                Square sq_p = static_cast<Square>((p - Eval::BonaPiece::fe_hand_end) % SQ_NB);
                sq_p = Mir(sq_p);
                p = static_cast<Eval::BonaPiece>(Eval::BonaPiece::fe_hand_end + piece_index * static_cast<int>(SQ_NB) + sq_p);
            }
        }
        // Map king square + piece to the final feature index.
        return static_cast<int>(Eval::BonaPiece::e_king) * static_cast<int>(sq_k) + static_cast<int>(p >= Eval::BonaPiece::e_king ? p - SQ_NB : p);
    }

    static std::pair<int, int> fill_features_sparse(const shogi::TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = *e.pos;
        Eval::BonaPiece* pieces = nullptr;
        if (color == Color::BLACK) {
            pieces = pos.eval_list()->piece_list_fb();
        }
        else {
            pieces = pos.eval_list()->piece_list_fw();
        }
        PieceNumber target = static_cast<PieceNumber>(PIECE_NUMBER_KING + color);
        auto sq_target_k = static_cast<Square>((pieces[target] - Eval::BonaPiece::f_king) % SQ_NB);

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int j = 0;
        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_NB; ++i)
        {
            auto p = pieces[i];
            values[j] = 1.0f;
            features[j] = make_index(sq_target_k, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

template <typename T, typename... Ts>
struct FeatureSet
{
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const shogi::TrainingDataEntry& e, int* features, float* values, Color color)
    {
        return T::fill_features_sparse(e, features, values, color);
    }
};

struct SparseBatch
{
    static constexpr bool IS_BATCH = true;

    template <typename... Ts>
    SparseBatch(FeatureSet<Ts...>, const std::vector<shogi::TrainingDataEntry>& entries)
    {
        num_inputs = FeatureSet<Ts...>::INPUTS;
        size = entries.size();
        is_white = new float[size];
        outcome = new float[size];
        score = new float[size];
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        white_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        layer_stack_indices = new int[size];

        num_active_white_features = 0;
        num_active_black_features = 0;
        max_active_features = FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;

        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white_values[i] = 0.0f;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black_values[i] = 0.0f;

        for (int i = 0; i < entries.size(); ++i)
        {
            fill_entry(FeatureSet<Ts...>{}, i, entries[i]);
        }
    }

    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    int num_active_white_features;
    int num_active_black_features;
    int max_active_features;
    int* white;
    int* black;
    float* white_values;
    float* black_values;
    int* layer_stack_indices;

    ~SparseBatch()
    {
        delete[] is_white;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
        delete[] white_values;
        delete[] black_values;
        delete[] layer_stack_indices;
    }

private:

    // Bucket selection based on progress (0..1) using 8 buckets.
    static constexpr int kLayerStacks = 8;
    static int stack_index_for_nnue(const Position& pos) {
        double progress = 0.0;
        if (TANUKI_PROGRESS_LOADED.load()) {
            progress = TANUKI_PROGRESS.Estimate(pos);
        }
        int idx = static_cast<int>(progress * static_cast<double>(kLayerStacks));
        if (idx < 0) idx = 0;
        if (idx >= kLayerStacks) idx = kLayerStacks - 1;
        return idx;
    }

    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const shogi::TrainingDataEntry& e)
    {
        is_white[i] = static_cast<float>(e.pos->side_to_move() == Color::BLACK);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        layer_stack_indices[i] = stack_index_for_nnue(*e.pos);
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    template <typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const shogi::TrainingDataEntry& e)
    {
        const int offset = i * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;
        num_active_white_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, white + offset, white_values + offset, Color::BLACK)
            .first;
        num_active_black_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, black + offset, black_values + offset, Color::WHITE)
            .first;
    }
};

struct AnyStream
{
    virtual ~AnyStream() = default;
};

template <typename StorageT>
struct Stream : AnyStream
{
    using StorageType = StorageT;

    Stream(int concurrency, const std::vector<std::string>& filenames, bool cyclic, std::function<bool(const shogi::TrainingDataEntry&)> skipPredicate) :
        m_stream(training_data::open_sfen_input_file_parallel(concurrency, filenames, cyclic, skipPredicate))
    {
    }

    virtual StorageT* next() = 0;

protected:
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

template <typename StorageT>
struct AsyncStream : Stream<StorageT>
{
    using BaseType = Stream<StorageT>;

    AsyncStream(int concurrency, const std::vector<std::string>& filenames, bool cyclic, std::function<bool(const shogi::TrainingDataEntry&)> skipPredicate) :
        BaseType(1, filenames, cyclic, skipPredicate)
    {
    }

    ~AsyncStream()
    {
        if (m_next.valid())
        {
            delete m_next.get();
        }
    }

protected:
    std::future<StorageT*> m_next;
};

template <typename FeatureSetT, typename StorageT>
struct FeaturedBatchStream : Stream<StorageT>
{
    static_assert(StorageT::IS_BATCH);

    using FeatureSet = FeatureSetT;
    using BaseType = Stream<StorageT>;

    static constexpr int num_feature_threads_per_reading_thread = 2;

    FeaturedBatchStream(int concurrency, const std::vector<std::string>& filenames, int batch_size, bool cyclic, std::function<bool(const shogi::TrainingDataEntry&)> skipPredicate) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filenames,
            cyclic,
            skipPredicate
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size)
    {
        m_stop_flag.store(false);

        auto worker = [this]()
            {
                std::vector<shogi::TrainingDataEntry> entries;
                entries.reserve(m_batch_size);

                while (!m_stop_flag.load())
                {
                    entries.clear();

                    {
                        std::unique_lock lock(m_stream_mutex);
                        BaseType::m_stream->fill(entries, m_batch_size);
                        if (entries.empty())
                        {
                            break;
                        }
                    }

                    auto batch = new StorageT(FeatureSet{}, entries);

                    {
                        std::unique_lock lock(m_batch_mutex);
                        m_batches_not_full.wait(lock, [this]() { return m_batches.size() < m_concurrency + 1 || m_stop_flag.load(); });

                        m_batches.emplace_back(batch);

                        lock.unlock();
                        m_batches_any.notify_one();
                    }

                }
                m_num_workers.fetch_sub(1);
                m_batches_any.notify_one();
            };

        const int num_feature_threads = std::max(
            1,
            concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread)
        );

        for (int i = 0; i < num_feature_threads; ++i)
        {
            m_workers.emplace_back(worker);

            // This cannot be done in the thread worker. We need
            // to have a guarantee that this is incremented, but if
            // we did it in the worker there's no guarantee
            // that it executed.
            m_num_workers.fetch_add(1);
        }
    }

    StorageT* next() override
    {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

        if (!m_batches.empty())
        {
            auto batch = m_batches.front();
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }
        return nullptr;
    }

    ~FeaturedBatchStream()
    {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for (auto& worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        for (auto& batch : m_batches)
        {
            delete batch;
        }
    }

private:
    int m_batch_size;
    int m_concurrency;
    std::deque<StorageT*> m_batches;
    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;

    std::vector<std::thread> m_workers;
};

struct DataloaderSkipConfig {
    bool filtered;
    int random_fen_skipping;
    bool wld_filtered;
    int early_fen_skipping;
    int simple_eval_skipping;
    int param_index;
};

std::function<bool(const shogi::TrainingDataEntry&)> make_skip_predicate(DataloaderSkipConfig config)
{
    return nullptr;
}

extern "C" {

    // changing the signature needs matching changes in data_loader/_native.py
    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(const char* feature_set_c, int concurrency, int num_files, const char* const* filenames, int batch_size, bool cyclic, DataloaderSkipConfig config)
    {
        auto initialize = []() {
            USI::init(Options);
            Tanuki::Progress::Initialize(Options);
            Bitboards::init();
            //Position::init();
            //Search::init();

            Threads.set(1);

            //Eval::init();

            is_ready();
            TANUKI_PROGRESS_LOADED = TANUKI_PROGRESS.Load();
            if (!TANUKI_PROGRESS_LOADED.load()) {
                fprintf(stderr, "Failed to load tanuki progress weights. Falling back to zero progress.\n");
            }
            };
        std::call_once(INITIALIZED, initialize);

        auto skipPredicate = make_skip_predicate(config);
        auto filenames_vec = std::vector<std::string>(filenames, filenames + num_files);

        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKA_hm")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKA_hm>, SparseBatch>(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
        }
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
    {
        delete stream;
    }

    EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
    {
        return stream->next();
    }

    EXPORT void CDECL destroy_sparse_batch(SparseBatch* e)
    {
        delete e;
    }

}

#if defined(BENCH)

/* benches
   compile and run with:
     g++ -std=c++20 -g3 -O3 -DNDEBUG -DBENCH -march=native training_data_loader.cpp && ./a.out /path/to/binpack
*/

#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

long long get_rchar_self() {
    std::ifstream io_file("/proc/self/io");
    std::string line;
    while (std::getline(io_file, line)) {
        if (line.rfind("rchar:", 0) == 0) {
            return std::stoll(line.substr(6));
        }
    }
    return -1; // Error or not found
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " file1 [file2 ...]\n";
        return 1;
    }

    const char** files = const_cast<const char**>(&argv[1]);
    int file_count = argc - 1;

#ifdef PGO_BUILD
    const int concurrency = 1;
#else
    //const int concurrency = std::thread::hardware_concurrency();
    const int concurrency = std::thread::hardware_concurrency();
#endif
    // some typical numbers, more skipping means more load
    const int batch_size = 16384;
    const bool cyclic = true;
    const DataloaderSkipConfig config = {
        .filtered = true,
        .random_fen_skipping = 3,
        .wld_filtered = true,
        .early_fen_skipping = 5,
        .simple_eval_skipping = 0,
        .param_index = 0
    };
    auto stream = create_sparse_batch_stream("HalfKA_hm", concurrency, file_count, files, batch_size, cyclic, config);

    auto t0 = std::chrono::high_resolution_clock::now();

#ifdef PGO_BUILD
    constexpr int iteration_count = 30;
#else
    constexpr int iteration_count = 6000;
#endif

    for (int i = 1; i <= iteration_count; ++i)
    {
        destroy_sparse_batch(stream->next());
        auto t1 = std::chrono::high_resolution_clock::now();
        if (i % 1 == 0)
        {
            double sec = (t1 - t0).count() / 1e9;
            long long bytes = get_rchar_self();
            std::cout << "\rIter:   " << std::fixed << std::setw(8) << i
                << "       Time(s): " << std::fixed << std::setw(8) << std::setprecision(3) << sec
                << "       MPos/s:   " << std::fixed << std::setw(8) << std::setprecision(3) << i * batch_size / (sec * 1000 * 1000)
                << "       It/s:    " << std::fixed << std::setw(8) << std::setprecision(1) << i / sec
                << "       MB/s:    " << std::fixed << std::setw(8) << std::setprecision(1) << bytes / (sec * 1024 * 1024)
                << "       B/pos:  " << std::fixed << std::setw(8) << std::setprecision(1) << bytes / (i * batch_size)
                << std::flush;
        }
    }
    std::cout << std::endl;

    return 0;
}

#endif
