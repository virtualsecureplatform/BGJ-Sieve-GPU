#include "../include/bounded_task_queue.h"
#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

int main() {
    bounded_task_queue<int> empty(0);
    empty.close(); int value;
    assert(!empty.pop(value));
    bounded_task_queue<int> full(1);
    full.push(7);
    bool rejected = false;
    try { full.push(8); } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    full.close(); assert(full.pop(value) && value == 7); assert(!full.pop(value));
    rejected = false;
    try { full.push(9); } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    for (int iteration = 0; iteration < 40; ++iteration) {
        bounded_task_queue<int> queue(10000);
        std::vector<std::atomic<int>> seen(10000);
        for (auto &count : seen) count = 0;
        std::vector<std::thread> workers;
        for (int i = 0; i < 8; ++i) workers.emplace_back([&] {
            int item;
            while (queue.pop(item)) { assert(item >= 0 && item < 10000); ++seen[item]; }
        });
        for (int i = 0; i < 10000; ++i) queue.push(i);
        queue.close();
        for (auto &worker : workers) worker.join();
        for (auto &count : seen) assert(count == 1);
    }
}
