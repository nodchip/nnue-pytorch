#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>
#include <mutex>
#include <thread>
#include <deque>
#include <random>

#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"

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

// ====== Shogi HalfKA_hm (hm + king-bucket + packed bona piece) ======

struct HalfKA_hm {
    // --- Shogi constants ---
    static constexpr int SQ_NB = 81;  // 9x9
    static constexpr int FILE_NB = 9;

    // "e_king" here is BonaPiece::e_king (start index of enemy king in BonaPiece space)
    // After packing enemy king by -SQ_NB, the packed range becomes [0, e_king-1]
    static constexpr int PIECE_INPUTS = static_cast<int>(Eval::BonaPiece::e_king);

    // King is bucketed into 9 ranks * 5 files (after half-mirror)
    static constexpr int KING_BUCKETS = 5 * FILE_NB; // 45

    static constexpr int INPUTS = KING_BUCKETS * PIECE_INPUTS;

    // Active = number of pieces in EvalList (typically 38), keep some slack
    static constexpr int MAX_ACTIVE_FEATURES = Eval::EvalList::MAX_LENGTH; // 40

    // Mirror by file for shogi squares (you likely already have Mir(sq) in your codebase)
    static inline Square mir_file(Square sq) {
        return Mir(sq);
    }

    // Convert oriented king square -> compact bucket [0..44]
    // - First: perspective orientation (black-view vs white-view)
    // - Then: half-mirror by file (6..9 -> 4..1)
    // - Finally: compress to rank*5 + file_m
    static inline int king_bucket(Color perspective, Square ksq) {
        // If your engine uses Inv() for white-view, apply it here.
        // In YaneuraOu EvalList, piece_list_fw() already uses Inv() for squares,
        // so for king-square we can just use the "from perspective" king square.
        // If you are taking king square directly from pos, you may need:
        //   if (perspective == WHITE) ksq = Inv(ksq);
        //
        // Here we assume ksq is already in "perspective view".

        int file = file_of(ksq); // 0..8
        int rank = rank_of(ksq); // 0..8

        // half-mirror: if file >= 5 (6..9ãÿ), mirror
        int file_m = (file >= 5) ? (8 - file) : file; // now 0..4

        return rank * 5 + file_m; // 0..44
    }

    // Pack BonaPiece:
    // - if king-side hm mirror is active, mirror board-square part only
    // - then pack enemy-king range into [0..e_king-1] by -SQ_NB
    static inline int pack_bonapiece(Eval::BonaPiece p, bool hm_mirror) {
        using BP = Eval::BonaPiece;
        const int ip = static_cast<int>(p);

        // Hand pieces must NOT be mirrored
        const int hand_end = static_cast<int>(BP::fe_hand_end);

        int pp = ip;
        if (hm_mirror && pp >= hand_end) {
            // board piece: (piece_index, sq) layout is: hand_end + piece_index*81 + sq
            const int rel = pp - hand_end;
            const int piece_index = rel / SQ_NB;
            const int sq = rel % SQ_NB;

            Square sq_p = static_cast<Square>(sq);
            sq_p = mir_file(sq_p);

            pp = hand_end + piece_index * SQ_NB + static_cast<int>(sq_p);
        }

        // pack enemy king to friend king plane
        const int e_king = static_cast<int>(BP::e_king);
        if (pp >= e_king)
            pp -= SQ_NB;

        return pp; // 0..(e_king-1)
    }

    // Feature index = king_bucket * PIECE_INPUTS + packed_bonapiece
    static inline int feature_index(Color perspective, Square ksq_persp, Eval::BonaPiece p) {
        int file = file_of(ksq_persp);
        bool hm_mirror = (file >= 5); // 6..9ãÿÇ»ÇÁÉ~ÉâÅ[
        int kb = king_bucket(perspective, ksq_persp);
        int pp = pack_bonapiece(p, hm_mirror);
        return kb * PIECE_INPUTS + pp;
    }

    static std::pair<int, int> fill_features_sparse(
        const shogi::TrainingDataEntry& e, int* features, float* values, Color perspective)
    {
        // Get BonaPiece list from EvalList in the chosen perspective
        auto* el = e.pos->eval_list();
        auto* pieces = (perspective == BLACK)
            ? el->piece_list_fb()
            : el->piece_list_fw();

        // Determine king square in the same perspective view
        // In YaneuraOu EvalList, king's BonaPiece is located at PIECE_NUMBER_KING + perspective
        const PieceNumber king_no = static_cast<PieceNumber>(PIECE_NUMBER_KING + perspective);
        Square ksq = static_cast<Square>((static_cast<int>(pieces[king_no]) - static_cast<int>(Eval::BonaPiece::f_king)) % SQ_NB);

        int j = 0;
        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_NB; ++i) {
            values[j] = 1.0f;
            features[j] = feature_index(perspective, ksq, pieces[i]);
            ++j;
        }
        return { j, INPUTS };
    }
};


// ====== Shogi HalfKA_hm^ (factorized) ======
// Adds "piece-only" factor: packed_bonapiece (same packing/mirror rule), independent of king bucket.
struct HalfKA_hmFactorized {
    static constexpr int SQ_NB = HalfKA_hm::SQ_NB;
    static constexpr int FILE_NB = HalfKA_hm::FILE_NB;

    static constexpr int PIECE_INPUTS = HalfKA_hm::PIECE_INPUTS;
    static constexpr int KING_BUCKETS = HalfKA_hm::KING_BUCKETS;

    static constexpr int BASE_INPUTS = HalfKA_hm::INPUTS;
    static constexpr int FACT_INPUTS = PIECE_INPUTS; // piece-only factor
    static constexpr int INPUTS = BASE_INPUTS + FACT_INPUTS;

    static constexpr int MAX_ACTIVE_FEATURES =
        HalfKA_hm::MAX_ACTIVE_FEATURES + HalfKA_hm::MAX_ACTIVE_FEATURES;

    static std::pair<int, int> fill_features_sparse(
        const shogi::TrainingDataEntry& e, int* features, float* values, Color perspective)
    {
        // 1) Base HalfKA_hm
        auto [start_j, base_inputs] = HalfKA_hm::fill_features_sparse(e, features, values, perspective);
        int j = start_j;

        // 2) Piece-only factor (offset = BASE_INPUTS)
        auto* el = e.pos->eval_list();
        auto* pieces = (perspective == BLACK)
            ? el->piece_list_fb()
            : el->piece_list_fw();

        const PieceNumber king_no = static_cast<PieceNumber>(PIECE_NUMBER_KING + perspective);
        Square ksq = static_cast<Square>((static_cast<int>(pieces[king_no]) - static_cast<int>(Eval::BonaPiece::f_king)) % SQ_NB);

        // Mirror rule depends on king file (same as base)
        int file = file_of(ksq);
        bool hm_mirror = (file >= 5);

        for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_NB; ++i) {
            values[j] = 1.0f;
            int pp = HalfKA_hm::pack_bonapiece(pieces[i], hm_mirror);
            features[j] = BASE_INPUTS + pp;
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
        psqt_indices = new int[size];
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
    int* psqt_indices;
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
        delete[] psqt_indices;
        delete[] layer_stack_indices;
    }

private:

    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const shogi::TrainingDataEntry& e)
    {
        is_white[i] = static_cast<float>(e.pos->side_to_move() == Color::BLACK);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        psqt_indices[i] = (e.pos->pieces().pop_count() - 1) / 4;
        layer_stack_indices[i] = psqt_indices[i];
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
        auto skipPredicate = make_skip_predicate(config);
        auto filenames_vec = std::vector<std::string>(filenames, filenames + num_files);

        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKA_hm")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKA_hm>, SparseBatch>(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKA_hm^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKA_hmFactorized>, SparseBatch>(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
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
    auto stream = create_sparse_batch_stream("HalfKAv2_hm^", concurrency, file_count, files, batch_size, cyclic, config);

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
