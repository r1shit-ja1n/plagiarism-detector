#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <map>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <limits>
using namespace std;

namespace PlagiarismChecker {

// ANSI escape codes for terminal output coloring
namespace Color {
    const std::string RESET   = "\033[0m";
    const std::string BOLD    = "\033[1m";
    const std::string RED     = "\033[31m";
    const std::string GREEN   = "\033[32m";
    const std::string YELLOW  = "\033[33m";
    const std::string BLUE    = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN    = "\033[36m";
    const std::string WHITE   = "\033[37m";
}

// Which string-matching algorithm to use when scoring similarity
enum class Algorithm {
    KMP,
    RABIN_KARP,
    LCS,
    ALL       // run all three and average the scores
};

// Global config passed around to every stage
struct Settings {
    int           ngram_size          = 3;
    Algorithm     preferred_algorithm = Algorithm::ALL;
    double        low_threshold       = 30.0;   // below this -> "likely original"
    double        high_threshold      = 70.0;   // above this -> "likely plagiarized"
    bool          use_color           = true;
    bool          remove_stopwords    = true;
    bool          verbose             = true;
    std::string   report_filename     = "plagiarism_report.txt";

    static std::string algorithmName(Algorithm a) {
        switch (a) {
            case Algorithm::KMP:        return "KMP";
            case Algorithm::RABIN_KARP: return "Rabin-Karp";
            case Algorithm::LCS:        return "LCS (DP)";
            case Algorithm::ALL:        return "ALL";
        }
        return "UNKNOWN";
    }

    void print() const {
        std::cout << Color::CYAN << "----- Current Settings -----\n" << Color::RESET;
        std::cout << "  N-gram size           : " << ngram_size << "\n";
        std::cout << "  Preferred algorithm   : " << algorithmName(preferred_algorithm) << "\n";
        std::cout << "  Low threshold (%)     : " << low_threshold << "\n";
        std::cout << "  High threshold (%)    : " << high_threshold << "\n";
        std::cout << "  Use ANSI color        : " << (use_color ? "yes" : "no") << "\n";
        std::cout << "  Remove stopwords      : " << (remove_stopwords ? "yes" : "no") << "\n";
        std::cout << "  Verbose mode          : " << (verbose ? "yes" : "no") << "\n";
        std::cout << "  Report filename       : " << report_filename << "\n";
        std::cout << Color::CYAN << "----------------------------\n" << Color::RESET;
    }
};

// Words that carry almost no meaning on their own and skew similarity scores
// if left in — "the", "and", "a", etc.
inline const std::unordered_set<std::string>& getStopwords() {
    static const std::unordered_set<std::string> STOPWORDS = {
        "a","an","the","and","or","but","if","while","with","of","at","by","for",
        "to","in","on","is","are","was","were","be","been","being","have","has",
        "had","do","does","did","not","no","this","that","these","those","i",
        "you","he","she","it","we","they","them","his","her","its","their","my",
        "your","our","me","him","us","what","which","who","whom","whose","when",
        "where","why","how","as","than","then","so","such","just","also","very",
        "can","could","should","would","may","might","must","shall","will","into",
        "from","about","over","under","again","further","once","here","there",
        "all","any","both","each","few","more","most","other","some","only","own",
        "same","too","s","t","d","ll","m","o","re","ve","y","up","down","out","off"
    };
    return STOPWORDS;
}

class Engine;

// ---------------------------------------------------------------------------
// Stage 1 — Naive similarity before any serious preprocessing
// Quick sanity-check metrics: word-level Jaccard and character-frequency overlap
// ---------------------------------------------------------------------------
namespace Stage1 {

    // splits on whitespace, no lowercasing or punctuation stripping
    inline std::vector<std::string> rawSplit(const std::string& s) {
        std::vector<std::string> out;
        std::istringstream iss(s);
        std::string token;
        while (iss >> token) out.push_back(token);
        return out;
    }

    // Jaccard similarity over the unique word sets of both strings.
    // |intersection| / |union|, scaled to [0, 100].
    inline double naiveWordSimilarity(const std::string& a, const std::string& b) {
        std::vector<std::string> wa = rawSplit(a);
        std::vector<std::string> wb = rawSplit(b);
        std::unordered_set<std::string> sa(wa.begin(), wa.end());
        std::unordered_set<std::string> sb(wb.begin(), wb.end());
        if (sa.empty() && sb.empty()) return 100.0;
        if (sa.empty() || sb.empty()) return 0.0;

        std::size_t inter = 0;
        for (const auto& w : sa) if (sb.count(w)) ++inter;
        std::size_t uni = sa.size() + sb.size() - inter;
        return (static_cast<double>(inter) / static_cast<double>(uni)) * 100.0;
    }

    // Compares how often each byte value appears in each string.
    // overlap = sum of min counts per character; total = sum of max counts.
    // Good for catching copy-paste even when word order changes.
    inline double naiveCharSimilarity(const std::string& a, const std::string& b) {
        if (a.empty() && b.empty()) return 100.0;
        if (a.empty() || b.empty()) return 0.0;
        vector<int> fa(256,0),fb(256,0);
        for (unsigned char c : a) ++fa[c];
        for (unsigned char c : b) ++fb[c];

        long long overlap = 0;
        long long total   = 0;
        for (int i = 0; i < 256; ++i) {
            overlap += std::min(fa[i], fb[i]);
            total   += std::max(fa[i], fb[i]);
        }
        if (total == 0) return 0.0;
        return (static_cast<double>(overlap) / static_cast<double>(total)) * 100.0;
    }

    inline void demonstrate(const std::string& a, const std::string& b) {
        std::cout << Color::BOLD << "[STAGE 1] Naive String Matching\n" << Color::RESET;
        std::cout << "  String A length: " << a.size() << "\n";
        std::cout << "  String B length: " << b.size() << "\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Naive word similarity : "
                  << naiveWordSimilarity(a, b) << " %\n";
        std::cout << "  Naive char similarity : "
                  << naiveCharSimilarity(a, b) << " %\n";
    }

}

// ---------------------------------------------------------------------------
// Stage 2 — File I/O helpers
// Everything that touches the filesystem lives here so the rest of the code
// doesn't have to deal with ifstream/ofstream directly.
// ---------------------------------------------------------------------------
namespace Stage2 {

    inline std::vector<std::string> readFileLines(const std::string& path) {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("Stage2::readFileLines: cannot open '" + path + "'");
        }
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(in, line)) lines.push_back(line);
        return lines;
    }

    inline std::string readFileWhole(const std::string& path) {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("Stage2::readFileWhole: cannot open '" + path + "'");
        }
        std::ostringstream oss;
        oss << in.rdbuf();
        return oss.str();
    }

    inline void writeFile(const std::string& path, const std::string& content) {
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("Stage2::writeFile: cannot open '" + path + "' for writing");
        }
        out << content;
    }

    inline std::string joinLines(const std::vector<std::string>& lines) {
        std::ostringstream oss;
        for (std::size_t i = 0; i < lines.size(); ++i) {
            oss << lines[i];
            if (i + 1 != lines.size()) oss << "\n";
        }
        return oss.str();
    }

    // Same as readFileWhole but returns a fallback string instead of throwing
    // when the file doesn't exist — useful for optional config files.
    inline std::string readFileWholeOr(const std::string& path,
                                       const std::string& fallback) {
        std::ifstream in(path);
        if (!in) return fallback;
        std::ostringstream oss;
        oss << in.rdbuf();
        return oss.str();
    }

}

// ---------------------------------------------------------------------------
// Stage 3 — Text preprocessing
// Raw document text -> cleaned token list ready for comparison.
// Pipeline: lowercase -> strip punctuation -> split -> remove stopwords
// ---------------------------------------------------------------------------
namespace Stage3 {

    inline std::string toLower(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (unsigned char c : s) out.push_back(static_cast<char>(std::tolower(c)));
        return out;
    }

    // Replaces every non-alphanumeric, non-space character with a space so that
    // "word," and "word" both tokenize to the same thing.
    inline std::string stripPunctuation(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (unsigned char c : s) {
            if (std::isalnum(c) || std::isspace(c)) out.push_back(static_cast<char>(c));
            else out.push_back(' ');
        }
        return out;
    }

    inline std::vector<std::string> tokenize(const std::string& s) {
        std::vector<std::string> tokens;
        std::istringstream iss(s);
        std::string tok;
        while (iss >> tok) tokens.push_back(tok);
        return tokens;
    }

    inline std::vector<std::string> removeStopwords(const std::vector<std::string>& toks) {
        const auto& sw = getStopwords();
        std::vector<std::string> out;
        out.reserve(toks.size());
        for (const auto& t : toks) {
            if (!sw.count(t)) out.push_back(t);
        }
        return out;
    }

    // Full preprocessing pipeline — result is a token list suitable for n-gram
    // or LCS comparison.
    inline std::vector<std::string> preprocess(const std::string& raw,
                                               const Settings& cfg) {
        std::string s = toLower(raw);
        s = stripPunctuation(s);
        std::vector<std::string> toks = tokenize(s);
        if (cfg.remove_stopwords) toks = removeStopwords(toks);
        return toks;
    }

    // Same as preprocess() but rejoins tokens into a single space-separated
    // string — needed when passing preprocessed text to the KMP/RK matchers.
    inline std::string preprocessAsString(const std::string& raw,
                                          const Settings& cfg) {
        std::vector<std::string> toks = preprocess(raw, cfg);
        std::ostringstream oss;
        for (std::size_t i = 0; i < toks.size(); ++i) {
            oss << toks[i];
            if (i + 1 != toks.size()) oss << ' ';
        }
        return oss.str();
    }

    inline void demonstrate(const std::string& raw, const Settings& cfg) {
        std::cout << Color::BOLD << "[STAGE 3] Text Preprocessing\n" << Color::RESET;
        std::cout << "  RAW   : " << raw.substr(0, std::min<std::size_t>(120, raw.size()))
                  << (raw.size() > 120 ? "..." : "") << "\n";
        auto toks = preprocess(raw, cfg);
        std::cout << "  TOKENS(" << toks.size() << "): ";
        for (std::size_t i = 0; i < toks.size() && i < 20; ++i) {
            std::cout << toks[i] << ' ';
        }
        if (toks.size() > 20) std::cout << "...";
        std::cout << "\n";
    }

}

// ---------------------------------------------------------------------------
// Stage 4 — N-gram Jaccard similarity
// Builds sets of overlapping n-word windows and measures how much they share.
// Jaccard = |A ∩ B| / |A ∪ B|
// ---------------------------------------------------------------------------
namespace Stage4 {

    // Slides a window of width n across the token list and builds every n-gram
    // as a single underscore-joined string, e.g. "quick_brown_fox".
    // Using a set automatically deduplicates repeated n-grams.
    inline std::unordered_set<std::string> buildNGrams(
            const std::vector<std::string>& tokens, int n) {
        std::unordered_set<std::string> grams;
        if (n <= 0 || tokens.size() < static_cast<std::size_t>(n)) return grams;
        grams.reserve(tokens.size());
        for (std::size_t i = 0; i + static_cast<std::size_t>(n) <= tokens.size(); ++i) {
            std::string g;
            for (int k = 0; k < n; ++k) {
                if (k) g.push_back('_');
                g += tokens[i + k];
            }
            grams.insert(std::move(g));
        }
        return grams;
    }

    // Always iterates over the smaller set to keep intersection counting fast.
    inline double jaccard(const std::unordered_set<std::string>& A,
                          const std::unordered_set<std::string>& B) {
        if (A.empty() && B.empty()) return 100.0;
        if (A.empty() || B.empty()) return 0.0;
        std::size_t inter = 0;

        const auto& small = (A.size() < B.size()) ? A : B;
        const auto& big   = (A.size() < B.size()) ? B : A;
        for (const auto& g : small) if (big.count(g)) ++inter;
        std::size_t uni = A.size() + B.size() - inter;
        return (static_cast<double>(inter) / static_cast<double>(uni)) * 100.0;
    }

    // End-to-end: preprocess both documents, build n-gram sets, return Jaccard %.
    inline double similarity(const std::string& docA,
                             const std::string& docB,
                             const Settings& cfg) {
        auto ta = Stage3::preprocess(docA, cfg);
        auto tb = Stage3::preprocess(docB, cfg);
        auto na = buildNGrams(ta, cfg.ngram_size);
        auto nb = buildNGrams(tb, cfg.ngram_size);
        return jaccard(na, nb);
    }

    inline void demonstrate(const std::string& a, const std::string& b,
                            const Settings& cfg) {
        std::cout << Color::BOLD << "[STAGE 4] N-Gram Jaccard Similarity (n="
                  << cfg.ngram_size << ")\n" << Color::RESET;
        double s = similarity(a, b, cfg);
        std::cout << std::fixed << std::setprecision(2)
                  << "  Jaccard similarity = " << s << " %\n";
    }

}

// ---------------------------------------------------------------------------
// Engine — the three core string-matching algorithms
// Each runXxx() method returns a Result with score, match count, timing, etc.
// ---------------------------------------------------------------------------
class Engine {
public:

    struct Result {
        std::string algorithm;
        double      score;
        long long   matches;      // n-grams from smaller doc found in larger doc
        std::size_t lcs_length;   // only populated by LCS; zero for KMP/RK
        long long   time_us;      // wall-clock time in microseconds
        std::string note;
    };

    // Builds the Longest Proper Prefix which is also Suffix (LPS) table.
    // This is the preprocessing step KMP needs to skip redundant comparisons.
    static std::vector<int> buildLPS(const std::string& pattern) {
        const std::size_t m = pattern.size();
        std::vector<int> lps(m, 0);
        std::size_t len = 0;
        for (std::size_t i = 1; i < m;) {
            if (pattern[i] == pattern[len]) {
                lps[i++] = static_cast<int>(++len);
            } else if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i++] = 0;
            }
        }
        return lps;
    }

    // Returns how many non-overlapping times pattern appears in text.
    // After a full match, jumps back using lps instead of restarting from i+1.
    static long long kmpCountMatches(const std::string& text,
                                     const std::string& pattern) {
        if (pattern.empty() || text.size() < pattern.size()) return 0;
        std::vector<int> lps = buildLPS(pattern);
        long long count = 0;
        std::size_t i = 0, j = 0;
        const std::size_t n = text.size(), m = pattern.size();
        while (i < n) {
            if (text[i] == pattern[j]) { ++i; ++j; }
            if (j == m) {
                ++count;
                j = static_cast<std::size_t>(lps[j - 1]);
            } else if (i < n && text[i] != pattern[j]) {
                if (j != 0) j = static_cast<std::size_t>(lps[j - 1]);
                else        ++i;
            }
        }
        return count;
    }

    // KMP-based n-gram matching:
    // Takes every n-gram from the smaller token list and searches for it in the
    // larger list (joined as a string). Score = matched n-grams / total n-grams.
    static Result runKMP(const std::vector<std::string>& tokensA,
                         const std::vector<std::string>& tokensB,
                         int ngram) {
        Result r{ "KMP", 0.0, 0, 0, 0, "Exact n-gram occurrence via LPS skip" };
        auto t0 = std::chrono::high_resolution_clock::now();

        const auto& small = (tokensA.size() <= tokensB.size()) ? tokensA : tokensB;
        const auto& big   = (tokensA.size() <= tokensB.size()) ? tokensB : tokensA;
        if (small.empty() || big.empty() ||
            small.size() < static_cast<std::size_t>(ngram)) {
            auto t1 = std::chrono::high_resolution_clock::now();
            r.time_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            return r;
        }

        // Flatten the larger token list into a single string so KMP can scan it.
        std::ostringstream oss;
        for (std::size_t i = 0; i < big.size(); ++i) {
            oss << big[i];
            if (i + 1 != big.size()) oss << ' ';
        }
        std::string bigStr = oss.str();

        long long total = 0, hits = 0;
        for (std::size_t i = 0; i + ngram <= small.size(); ++i) {
            // Build the n-gram pattern string for this window position
            std::string pat;
            for (int k = 0; k < ngram; ++k) {
                if (k) pat.push_back(' ');
                pat += small[i + k];
            }
            ++total;
            if (kmpCountMatches(bigStr, pat) > 0) ++hits;
        }
        if (total > 0) r.score = 100.0 * static_cast<double>(hits) /
                                 static_cast<double>(total);
        r.matches = hits;

        auto t1 = std::chrono::high_resolution_clock::now();
        r.time_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        return r;
    }

    // Returns how many non-overlapping times pattern appears in text using a
    // rolling polynomial hash. Hash collisions are resolved with a character
    // comparison to avoid false positives.
    static long long rabinKarpCountMatches(const std::string& text,
                                           const std::string& pattern) {
        const std::size_t n = text.size(), m = pattern.size();
        if (m == 0 || n < m) return 0;

        const std::uint64_t BASE = 257ULL;
        const std::uint64_t MOD  = 1000000007ULL;
        std::uint64_t patternHash = 0, windowHash = 0, high = 1;

        // high = BASE^(m-1) mod MOD — used to remove the leading character's
        // contribution when sliding the window forward by one position.
        for (std::size_t i = 0; i + 1 < m; ++i) high = (high * BASE) % MOD;
        for (std::size_t i = 0; i < m; ++i) {
            patternHash = (patternHash * BASE + static_cast<unsigned char>(pattern[i])) % MOD;
            windowHash  = (windowHash  * BASE + static_cast<unsigned char>(text[i]))    % MOD;
        }

        long long count = 0;
        for (std::size_t i = 0; i + m <= n; ++i) {
            if (patternHash == windowHash) {
                // Hash matched — confirm with a character-by-character check
                bool match = true;
                for (std::size_t k = 0; k < m; ++k) {
                    if (text[i + k] != pattern[k]) { match = false; break; }
                }
                if (match) ++count;
            }
            if (i + m < n) {
                // Slide the window: drop text[i], add text[i+m]
                std::uint64_t leading = (static_cast<unsigned char>(text[i]) * high) % MOD;
                windowHash = (windowHash + MOD - leading) % MOD;
                windowHash = (windowHash * BASE +
                              static_cast<unsigned char>(text[i + m])) % MOD;
            }
        }
        return count;
    }

    // Same n-gram matching logic as runKMP but uses the rolling hash to find
    // matches instead of the LPS table.
    static Result runRabinKarp(const std::vector<std::string>& tokensA,
                               const std::vector<std::string>& tokensB,
                               int ngram) {
        Result r{ "Rabin-Karp", 0.0, 0, 0, 0,
                  "Rolling-hash O(1) window updates" };
        auto t0 = std::chrono::high_resolution_clock::now();

        const auto& small = (tokensA.size() <= tokensB.size()) ? tokensA : tokensB;
        const auto& big   = (tokensA.size() <= tokensB.size()) ? tokensB : tokensA;
        if (small.empty() || big.empty() ||
            small.size() < static_cast<std::size_t>(ngram)) {
            auto t1 = std::chrono::high_resolution_clock::now();
            r.time_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            return r;
        }

        std::ostringstream oss;
        for (std::size_t i = 0; i < big.size(); ++i) {
            oss << big[i];
            if (i + 1 != big.size()) oss << ' ';
        }
        std::string bigStr = oss.str();

        long long total = 0, hits = 0;
        for (std::size_t i = 0; i + ngram <= small.size(); ++i) {
            std::string pat;
            for (int k = 0; k < ngram; ++k) {
                if (k) pat.push_back(' ');
                pat += small[i + k];
            }
            ++total;
            if (rabinKarpCountMatches(bigStr, pat) > 0) ++hits;
        }
        if (total > 0) r.score = 100.0 * static_cast<double>(hits) /
                                 static_cast<double>(total);
        r.matches = hits;

        auto t1 = std::chrono::high_resolution_clock::now();
        r.time_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        return r;
    }

    // LCS via DP on token sequences.
    // Uses two rolling rows instead of the full (n x m) table to keep memory O(m).
    // Score = 2 * LCS_length / (|A| + |B|)  — a.k.a. the Sørensen–Dice coefficient.
    static std::size_t lcsLength(const std::vector<std::string>& A,
                                 const std::vector<std::string>& B) {
        if (A.empty() || B.empty()) return 0;

        // Always iterate over the longer sequence in the outer loop so the inner
        // vector (prev/curr) stays as short as possible.
        const std::vector<std::string>* longer  = &A;
        const std::vector<std::string>* shorter = &B;
        if (A.size() < B.size()) std::swap(longer, shorter);

        const std::size_t n = longer->size();
        const std::size_t m = shorter->size();
        std::vector<std::size_t> prev(m + 1, 0), curr(m + 1, 0);
        for (std::size_t i = 1; i <= n; ++i) {
            for (std::size_t j = 1; j <= m; ++j) {
                if ((*longer)[i - 1] == (*shorter)[j - 1])
                    curr[j] = prev[j - 1] + 1;     // tokens match: extend the subsequence
                else
                    curr[j] = std::max(prev[j], curr[j - 1]);  // take the best so far
            }
            std::swap(prev, curr);
            std::fill(curr.begin(), curr.end(), 0);
        }
        return prev[m];
    }

    static Result runLCS(const std::vector<std::string>& tokensA,
                         const std::vector<std::string>& tokensB) {
        Result r{ "LCS", 0.0, 0, 0, 0,
                  "Dynamic Programming on token sequences" };
        auto t0 = std::chrono::high_resolution_clock::now();
        std::size_t L = lcsLength(tokensA, tokensB);
        r.lcs_length = L;
        std::size_t total = tokensA.size() + tokensB.size();
        if (total > 0) r.score = 100.0 * 2.0 * static_cast<double>(L) /
                                 static_cast<double>(total);
        auto t1 = std::chrono::high_resolution_clock::now();
        r.time_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        return r;
    }

    // Entry point: preprocesses both documents then dispatches to the
    // algorithm(s) selected in cfg. Returns one Result per algorithm run.
    static std::vector<Result> run(const std::string& docA,
                                   const std::string& docB,
                                   const Settings& cfg) {
        std::vector<std::string> ta = Stage3::preprocess(docA, cfg);
        std::vector<std::string> tb = Stage3::preprocess(docB, cfg);
        std::vector<Result> results;
        const int n = std::max(1, cfg.ngram_size);

        switch (cfg.preferred_algorithm) {
            case Algorithm::KMP:
                results.push_back(runKMP(ta, tb, n));
                break;
            case Algorithm::RABIN_KARP:
                results.push_back(runRabinKarp(ta, tb, n));
                break;
            case Algorithm::LCS:
                results.push_back(runLCS(ta, tb));
                break;
            case Algorithm::ALL:
            default:
                results.push_back(runKMP(ta, tb, n));
                results.push_back(runRabinKarp(ta, tb, n));
                results.push_back(runLCS(ta, tb));
                break;
        }
        return results;
    }
};

// ---------------------------------------------------------------------------
// Stage 6 — Multi-file comparison
// Scores a target document against an entire reference library and ranks results.
// ---------------------------------------------------------------------------
namespace Stage6 {

    struct Match {
        std::string label;
        double      similarity;
    };

    // Computes Jaccard n-gram similarity between the target and each reference,
    // then returns the list sorted highest-to-lowest.
    inline std::vector<Match> rankAgainstMany(
            const std::string& target,
            const std::vector<std::pair<std::string, std::string>>& refs,
            const Settings& cfg) {
        std::vector<Match> out;
        out.reserve(refs.size());
        for (const auto& [label, content] : refs) {
            double s = Stage4::similarity(target, content, cfg);
            out.push_back({ label, s });
        }
        std::sort(out.begin(), out.end(),
                  [](const Match& a, const Match& b) {
                      return a.similarity > b.similarity;
                  });
        return out;
    }

    // Convenience wrapper — just returns the single most similar reference.
    inline Match findMostSimilar(
            const std::string& target,
            const std::vector<std::pair<std::string, std::string>>& refs,
            const Settings& cfg) {
        if (refs.empty()) return { "(none)", 0.0 };
        auto ranked = rankAgainstMany(target, refs, cfg);
        return ranked.front();
    }

}

// ---------------------------------------------------------------------------
// Stage 7 — Console output
// Formatting helpers for scores, tables, and the highlighted overlap view.
// ---------------------------------------------------------------------------
namespace Stage7 {

    // Returns the ANSI color code that corresponds to a similarity score
    // relative to the configured thresholds.
    inline std::string scoreColor(double s, const Settings& cfg) {
        if (!cfg.use_color) return "";
        if (s >= cfg.high_threshold) return Color::RED;
        if (s >= cfg.low_threshold)  return Color::YELLOW;
        return Color::GREEN;
    }

    inline std::string resetColor(const Settings& cfg) {
        return cfg.use_color ? Color::RESET : "";
    }

    inline void printResultRow(const Engine::Result& r, const Settings& cfg) {
        std::cout << "  " << std::left << std::setw(14) << r.algorithm
                  << " | score: " << std::right << std::setw(6)
                  << std::fixed << std::setprecision(2) << r.score << " %"
                  << " | time: " << std::setw(8) << r.time_us << " us"
                  << " | matches: " << std::setw(6) << r.matches
                  << " | LCS|: " << std::setw(5) << r.lcs_length
                  << "  (" << r.note << ")\n";
        (void)cfg;
    }

    inline void printAlgorithmTable(const std::vector<Engine::Result>& results,
                                    const Settings& cfg) {
        std::cout << Color::BOLD
                  << "+--------------------------------------------------------"
                     "------------------------------------+\n"
                  << "| Algorithm    |  Score (%)  |  Time (us)  |  Matches  |"
                     "  LCS Len  |  Notes                  |\n"
                  << "+--------------+-------------+-------------+-----------+"
                     "-----------+-------------------------+\n"
                  << Color::RESET;
        for (const auto& r : results) {
            std::cout << "| " << std::left << std::setw(12) << r.algorithm
                      << " | " << std::right << std::setw(11) << std::fixed
                      << std::setprecision(2) << r.score
                      << " | " << std::setw(11) << r.time_us
                      << " | " << std::setw(9) << r.matches
                      << " | " << std::setw(9) << r.lcs_length
                      << " | " << std::left << std::setw(23)
                      << r.note.substr(0, 23) << " |\n";
        }
        std::cout << Color::BOLD
                  << "+--------------+-------------+-------------+-----------+"
                     "-----------+-------------------------+\n"
                  << Color::RESET;
        (void)cfg;
    }

    // Marks tokens in document B that appear in any matching n-gram shared with A.
    // Output is a space-separated token string with >>MATCH<< ... <<END<< tags
    // wrapping every contiguous run of matched tokens.
    inline std::string highlightCommonNGrams(const std::string& docA,
                                             const std::string& docB,
                                             const Settings& cfg) {
        auto ta = Stage3::preprocess(docA, cfg);
        auto tb = Stage3::preprocess(docB, cfg);
        auto na = Stage4::buildNGrams(ta, cfg.ngram_size);

        std::ostringstream oss;
        const int n = cfg.ngram_size;
        if (static_cast<int>(tb.size()) < n) return docB;

        // mark[i] = true if token i in tb is part of at least one shared n-gram
        std::vector<bool> mark(tb.size(), false);
        for (std::size_t i = 0; i + n <= tb.size(); ++i) {
            std::string g;
            for (int k = 0; k < n; ++k) {
                if (k) g.push_back('_');
                g += tb[i + k];
            }
            if (na.count(g)) {
                for (int k = 0; k < n; ++k) mark[i + k] = true;
            }
        }

        // Walk the mark array and insert the boundary tags whenever we enter or
        // exit a run of matched tokens.
        bool inMatch = false;
        for (std::size_t i = 0; i < tb.size(); ++i) {
            if (mark[i] && !inMatch) { oss << ">>MATCH<<"; inMatch = true; }
            if (!mark[i] && inMatch) { oss << "<<END<< ";  inMatch = false; }
            oss << tb[i] << ' ';
        }
        if (inMatch) oss << "<<END<<";
        return oss.str();
    }

    inline void printSummary(const std::string& labelA, const std::string& labelB,
                             double overall,
                             const std::vector<Engine::Result>& results,
                             const Settings& cfg) {
        std::cout << "\n" << Color::BOLD << "===== Plagiarism Summary =====\n"
                  << Color::RESET;
        std::cout << "  Document A : " << labelA << "\n";
        std::cout << "  Document B : " << labelB << "\n";
        std::cout << "  Overall Similarity: "
                  << scoreColor(overall, cfg) << std::fixed
                  << std::setprecision(2) << overall << " %"
                  << resetColor(cfg) << "\n";
        printAlgorithmTable(results, cfg);
    }

}

// ---------------------------------------------------------------------------
// Stage 8 — Report generation
// Writes a plain-text report file with all scores and a final verdict.
// ---------------------------------------------------------------------------
namespace Stage8 {

    inline std::string verdictFor(double overall, const Settings& cfg) {
        if (overall < cfg.low_threshold)        return "LOW (likely original)";
        if (overall <= cfg.high_threshold)      return "MODERATE (review recommended)";
        return "HIGH (likely plagiarized)";
    }

    inline void generateReport(const std::string& labelA,
                               const std::string& labelB,
                               double overall,
                               const std::vector<Engine::Result>& results,
                               const Settings& cfg) {
        std::ostringstream oss;
        oss << "================================================================\n";
        oss << "                   PLAGIARISM DETECTION REPORT                  \n";
        oss << "================================================================\n";
        oss << "Document A           : " << labelA << "\n";
        oss << "Document B           : " << labelB << "\n";
        oss << "N-gram size          : " << cfg.ngram_size << "\n";
        oss << "Stopword removal     : " << (cfg.remove_stopwords ? "yes" : "no") << "\n";
        oss << "Low threshold (%)    : " << cfg.low_threshold << "\n";
        oss << "High threshold (%)   : " << cfg.high_threshold << "\n";
        oss << "----------------------------------------------------------------\n";
        oss << std::fixed << std::setprecision(2);
        oss << "Overall Similarity   : " << overall << " %\n";
        oss << "----------------------------------------------------------------\n";
        oss << "Algorithm Breakdown:\n";
        for (const auto& r : results) {
            oss << "  - " << std::left << std::setw(12) << r.algorithm
                << "  score=" << std::setw(7) << r.score << "%"
                << "  time=" << std::setw(8) << r.time_us << "us"
                << "  matches=" << r.matches
                << "  lcs=" << r.lcs_length << "\n";
        }
        oss << "----------------------------------------------------------------\n";
        oss << "FINAL VERDICT        : " << verdictFor(overall, cfg) << "\n";
        oss << "================================================================\n";

        Stage2::writeFile(cfg.report_filename, oss.str());
        if (cfg.verbose) {
            std::cout << Color::GREEN
                      << "  [Report] Written to '" << cfg.report_filename << "'.\n"
                      << Color::RESET;
        }
    }

}

// Average all algorithm scores into a single number for threshold comparison.
inline double aggregateScore(const std::vector<Engine::Result>& results) {
    if (results.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& r : results) sum += r.score;
    return sum / static_cast<double>(results.size());
}

// ---------------------------------------------------------------------------
// Built-in sample documents used for demos and automated tests
// ---------------------------------------------------------------------------
namespace SampleData {

    inline const std::string DOC_ORIGINAL = R"(
        The quick brown fox jumps over the lazy dog. Algorithms in computer science are
        precise step-by-step procedures used to solve well-defined computational problems.
        Among the most studied are sorting algorithms, graph traversal algorithms, and
        string matching algorithms such as the Knuth-Morris-Pratt algorithm and the
        Rabin-Karp algorithm. The longest common subsequence problem is a classic example
        of dynamic programming.  Plagiarism detection often combines lexical preprocessing
        with both substring matching and subsequence-based comparisons in order to
        achieve robust similarity scoring across a wide variety of document styles.
    )";

    // Same ideas as DOC_ORIGINAL but reworded — tests whether the system catches
    // paraphrased plagiarism that wouldn't be caught by exact-match alone.
    inline const std::string DOC_PARAPHRASE = R"(
        The fast brown fox leaps over a lazy dog. In computer science, algorithms are
        exact step-by-step procedures designed to solve clearly defined problems. Some
        of the most widely studied are sorting algorithms, graph traversal techniques,
        and string searching algorithms like Knuth-Morris-Pratt and Rabin-Karp. The
        longest common subsequence problem is a famous illustration of dynamic
        programming. Plagiarism detection systems usually combine lexical preprocessing
        with substring and subsequence comparison to provide robust similarity scoring.
    )";

    // Completely different topic — should score very low against DOC_ORIGINAL.
    inline const std::string DOC_UNRELATED = R"(
        Photosynthesis is the biological process by which green plants, algae, and
        certain bacteria convert sunlight, carbon dioxide, and water into glucose
        and oxygen. The chloroplasts inside plant cells contain chlorophyll, the
        pigment responsible for absorbing light energy. This process is fundamental
        to life on Earth because it forms the base of most food chains and is
        responsible for the oxygen content of our planet's atmosphere.
    )";

    // Two original sentences wrapped around a verbatim lift from DOC_ORIGINAL —
    // tests whether partial plagiarism is detected even with surrounding original text.
    inline const std::string DOC_PARTIAL_COPY = R"(
        My essay topic is about animals and weather, but I will also mention that
        the quick brown fox jumps over the lazy dog. Algorithms in computer science
        are precise step-by-step procedures used to solve well-defined computational
        problems. After this borrowed paragraph I will continue with my own original
        observations about local wildlife and seasonal patterns in my hometown.
    )";

    inline std::vector<std::pair<std::string, std::string>> referenceLibrary() {
        return {
            { "Original.txt",     DOC_ORIGINAL     },
            { "Paraphrase.txt",   DOC_PARAPHRASE   },
            { "Unrelated.txt",    DOC_UNRELATED    },
            { "PartialCopy.txt",  DOC_PARTIAL_COPY }
        };
    }

}

}

using namespace PlagiarismChecker;

// Flushes any leftover characters from stdin after reading a value with >>.
static void clearStdin() {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

static int readInt(const std::string& prompt, int lo, int hi) {
    while (true) {
        std::cout << prompt;
        int x;
        if (std::cin >> x && x >= lo && x <= hi) {
            clearStdin();
            return x;
        }
        std::cout << "  Invalid; please enter an integer in ["
                  << lo << "," << hi << "].\n";
        clearStdin();
    }
}

static double readDouble(const std::string& prompt, double lo, double hi) {
    while (true) {
        std::cout << prompt;
        double x;
        if (std::cin >> x && x >= lo && x <= hi) {
            clearStdin();
            return x;
        }
        std::cout << "  Invalid; please enter a number in ["
                  << lo << "," << hi << "].\n";
        clearStdin();
    }
}

struct TestPair {
    std::string labelA, labelB;
    std::string docA,   docB;
};

// Holds everything produced by one pair comparison so callers can either
// print it immediately or pass it to the report generator.
struct PairOutcome {
    std::string                 labelA, labelB;
    double                      overall;
    std::vector<Engine::Result> results;
};

static std::vector<TestPair> buildTestSuite() {
    return {
        { "Original.txt",   "Paraphrase.txt",
          SampleData::DOC_ORIGINAL, SampleData::DOC_PARAPHRASE },
        { "Original.txt",   "Unrelated.txt",
          SampleData::DOC_ORIGINAL, SampleData::DOC_UNRELATED },
        { "Original.txt",   "PartialCopy.txt",
          SampleData::DOC_ORIGINAL, SampleData::DOC_PARTIAL_COPY },
        { "Paraphrase.txt", "PartialCopy.txt",
          SampleData::DOC_PARAPHRASE, SampleData::DOC_PARTIAL_COPY }
    };
}

// Runs the full Engine pipeline on a pair and optionally prints the summary
// + highlighted overlap to stdout. Returns the outcome for further use.
static PairOutcome runOnePair(const TestPair& tp, const Settings& cfg,
                              bool printToConsole = true) {
    PairOutcome out{ tp.labelA, tp.labelB, 0.0, {} };
    out.results = Engine::run(tp.docA, tp.docB, cfg);
    out.overall = aggregateScore(out.results);

    if (printToConsole) {
        Stage7::printSummary(tp.labelA, tp.labelB, out.overall, out.results, cfg);
        std::string highlighted =
            Stage7::highlightCommonNGrams(tp.docA, tp.docB, cfg);
        std::cout << Color::BOLD << "  -- Highlighted overlap (B vs A) --\n"
                  << Color::RESET
                  << "  "
                  << highlighted.substr(0, std::min<std::size_t>(500,
                                                                 highlighted.size()))
                  << (highlighted.size() > 500 ? " ..." : "")
                  << "\n";
    }
    return out;
}

static void runFullTestSuite(const Settings& cfg) {
    std::cout << Color::BOLD << Color::MAGENTA
              << "\n========== Running Full Automated Test Suite ==========\n"
              << Color::RESET;
    auto suite = buildTestSuite();
    int idx = 1;
    for (const auto& tp : suite) {
        std::cout << "\n--- Test " << idx++ << ": "
                  << tp.labelA << " vs " << tp.labelB << " ---\n";
        runOnePair(tp, cfg, true);
    }
    std::cout << Color::BOLD << Color::MAGENTA
              << "\n=========== Test Suite Complete ===========\n"
              << Color::RESET;
}

static void demoStage1And3And4(const Settings& cfg) {
    std::cout << "\n" << Color::BOLD << "----- Stage 1 / 3 / 4 Demo -----\n"
              << Color::RESET;
    Stage1::demonstrate(SampleData::DOC_ORIGINAL, SampleData::DOC_PARAPHRASE);
    std::cout << "\n";
    Stage3::demonstrate(SampleData::DOC_ORIGINAL, cfg);
    std::cout << "\n";
    Stage4::demonstrate(SampleData::DOC_ORIGINAL, SampleData::DOC_PARAPHRASE, cfg);
}

static void multiFileDemo(const Settings& cfg) {
    std::cout << "\n" << Color::BOLD << "----- STAGE 6: Multi-File Comparison -----\n"
              << Color::RESET;
    auto refs = SampleData::referenceLibrary();
    std::string target = SampleData::DOC_PARTIAL_COPY;
    auto ranked = Stage6::rankAgainstMany(target, refs, cfg);
    std::cout << "  Target: PartialCopy.txt\n";
    std::cout << "  Ranked by Jaccard (n=" << cfg.ngram_size << "):\n";
    for (const auto& m : ranked) {
        std::cout << "    " << std::left << std::setw(20) << m.label
                  << "  " << std::fixed << std::setprecision(2)
                  << m.similarity << " %\n";
    }
    auto top = Stage6::findMostSimilar(target, refs, cfg);
    std::cout << Color::BOLD << "  Most similar reference: " << Color::RESET
              << top.label << " (" << std::fixed << std::setprecision(2)
              << top.similarity << " %)\n";
}

// Runs all three algorithms on the same pair and prints the table so the user
// can see timing and score differences side-by-side.
static void timeComparisonDemo(const Settings& cfg) {
    std::cout << "\n" << Color::BOLD
              << "----- STAGE 5: Algorithm Time Comparison -----\n"
              << Color::RESET;
    Settings local = cfg;
    local.preferred_algorithm = Algorithm::ALL;
    auto results = Engine::run(SampleData::DOC_ORIGINAL,
                               SampleData::DOC_PARAPHRASE, local);
    Stage7::printAlgorithmTable(results, local);
}

static void generateSampleReport(const Settings& cfg) {
    TestPair tp{ "Original.txt", "PartialCopy.txt",
                 SampleData::DOC_ORIGINAL, SampleData::DOC_PARTIAL_COPY };
    auto outcome = runOnePair(tp, cfg, false);
    Stage8::generateReport(outcome.labelA, outcome.labelB,
                           outcome.overall, outcome.results, cfg);
    std::cout << Color::GREEN
              << "Sample report generated for '" << outcome.labelA
              << "' vs '" << outcome.labelB << "'.\n" << Color::RESET;
    std::cout << "Verdict: "
              << Stage8::verdictFor(outcome.overall, cfg) << "\n";
}

static void settingsPanel(Settings& cfg) {
    while (true) {
        std::cout << "\n" << Color::BOLD << "----- Settings Panel -----\n"
                  << Color::RESET;
        cfg.print();
        std::cout << "  1) Change n-gram size\n";
        std::cout << "  2) Change preferred algorithm\n";
        std::cout << "  3) Change low threshold\n";
        std::cout << "  4) Change high threshold\n";
        std::cout << "  5) Toggle ANSI color\n";
        std::cout << "  6) Toggle stopword removal\n";
        std::cout << "  7) Toggle verbose mode\n";
        std::cout << "  8) Change report filename\n";
        std::cout << "  0) Back\n";
        int c = readInt("  > Choice: ", 0, 8);
        if (c == 0) return;
        switch (c) {
            case 1: cfg.ngram_size =
                        readInt("  New n-gram size [1..8]: ", 1, 8); break;
            case 2: {
                std::cout << "    1) KMP   2) Rabin-Karp   3) LCS   4) ALL\n";
                int a = readInt("  Algorithm: ", 1, 4);
                cfg.preferred_algorithm =
                    (a == 1) ? Algorithm::KMP        :
                    (a == 2) ? Algorithm::RABIN_KARP :
                    (a == 3) ? Algorithm::LCS        :
                               Algorithm::ALL;
                break;
            }
            case 3: cfg.low_threshold =
                        readDouble("  New low threshold [0..100]: ", 0, 100); break;
            case 4: cfg.high_threshold =
                        readDouble("  New high threshold [0..100]: ", 0, 100); break;
            case 5: cfg.use_color        = !cfg.use_color;        break;
            case 6: cfg.remove_stopwords = !cfg.remove_stopwords; break;
            case 7: cfg.verbose          = !cfg.verbose;          break;
            case 8: {
                std::cout << "  New report filename: ";
                std::string fn;
                std::getline(std::cin, fn);
                if (!fn.empty()) cfg.report_filename = fn;
                break;
            }
        }
    }
}

// Reads two files from paths the user types in, falls back to the built-in
// sample documents if either path can't be opened.
static void compareTwoFilesFromDisk(const Settings& cfg) {
    std::cout << "\n" << Color::BOLD
              << "----- STAGE 2: Compare Two .txt Files (from disk) -----\n"
              << Color::RESET;
    std::cout << "  Path of File A: ";
    std::string pathA, pathB;
    std::getline(std::cin, pathA);
    std::cout << "  Path of File B: ";
    std::getline(std::cin, pathB);

    std::string a, b;
    try {
        a = Stage2::readFileWhole(pathA);
        b = Stage2::readFileWhole(pathB);
    } catch (const std::exception& e) {
        std::cout << Color::RED << "  Error: " << e.what() << Color::RESET << "\n";
        std::cout << "  Falling back to hardcoded sample documents.\n";
        a = SampleData::DOC_ORIGINAL;
        b = SampleData::DOC_PARAPHRASE;
        pathA = "Original.txt(sample)";
        pathB = "Paraphrase.txt(sample)";
    }
    TestPair tp{ pathA, pathB, a, b };
    auto outcome = runOnePair(tp, cfg, true);
    Stage8::generateReport(outcome.labelA, outcome.labelB,
                           outcome.overall, outcome.results, cfg);
}

static void printBanner() {
    std::cout << Color::BOLD << Color::CYAN
              << "================================================================\n"
              << "       PLAGIARISM DETECTION SYSTEM - DAA EDITION (C++17)        \n"
              << "    Stages: 1) Naive  2) File I/O  3) Preproc  4) N-grams       \n"
              << "            5) KMP/RK/LCS  6) Multi-file  7) Console            \n"
              << "            8) Report  9) Settings  10) Interactive             \n"
              << "================================================================\n"
              << Color::RESET;
}

static void printMenu() {
    std::cout << "\n" << Color::BOLD << "----- MAIN MENU -----\n" << Color::RESET;
    std::cout << "  1) Run full automated test suite\n";
    std::cout << "  2) Generate sample plagiarism report\n";
    std::cout << "  3) View algorithm time comparison\n";
    std::cout << "  4) Demo Stages 1, 3, 4 individually\n";
    std::cout << "  5) Multi-file comparison demo (Stage 6)\n";
    std::cout << "  6) Compare two .txt files from disk (Stage 2)\n";
    std::cout << "  7) Open Settings Panel (Stage 9)\n";
    std::cout << "  0) Exit\n";
}

int main() {
    Settings cfg;
    printBanner();
    cfg.print();

    while (true) {
        printMenu();
        int choice = readInt("  > Choice: ", 0, 7);
        switch (choice) {
            case 0:
                std::cout << Color::BOLD << "Goodbye!\n" << Color::RESET;
                return 0;
            case 1: runFullTestSuite(cfg);                    break;
            case 2: generateSampleReport(cfg);                break;
            case 3: timeComparisonDemo(cfg);                  break;
            case 4: demoStage1And3And4(cfg);                  break;
            case 5: multiFileDemo(cfg);                       break;
            case 6: compareTwoFilesFromDisk(cfg);             break;
            case 7: settingsPanel(cfg);                       break;
            default: break;
        }
    }
    return 0;
}