#!/usr/bin/env python3
"""Check that two square integer bases have equal determinants up to sign."""

import argparse


def load_basis(path):
    rows = []
    with open(path) as stream:
        for line in stream:
            line = line.strip()
            if line.startswith("[["):
                line = line[2:]
            elif line.startswith("["):
                line = line[1:]
            if line.endswith("]]" ):
                line = line[:-2]
            elif line.endswith("]"):
                line = line[:-1]
            if line:
                rows.append([int(value) for value in line.split()])
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError("basis must be a nonempty square integer matrix: {}".format(path))
    return rows


def determinant_mod(matrix, prime):
    work = [[value % prime for value in row] for row in matrix]
    determinant = 1
    for column in range(len(work)):
        pivot = next((row for row in range(column, len(work)) if work[row][column]), None)
        if pivot is None:
            return 0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = -determinant
        pivot_value = work[column][column]
        determinant = determinant * pivot_value % prime
        inverse = pow(pivot_value, prime - 2, prime)
        for row in range(column + 1, len(work)):
            if work[row][column] == 0:
                continue
            quotient = work[row][column] * inverse % prime
            work[row][column:] = [
                (left - quotient * right) % prime
                for left, right in zip(work[row][column:], work[column][column:])
            ]
    return determinant % prime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    reference = load_basis(args.reference)
    candidate = load_basis(args.candidate)
    if len(reference) != len(candidate):
        raise SystemExit("basis dimensions differ")
    for prime in (1000000007, 1000000009, 998244353, 2147483647):
        expected = determinant_mod(reference, prime)
        actual = determinant_mod(candidate, prime)
        if actual != expected and actual != (-expected) % prime:
            raise SystemExit(
                "determinant mismatch modulo {}: reference={} candidate={}".format(
                    prime, expected, actual
                )
            )
        print("determinant_check prime={} reference={} candidate={} status=ok".format(
            prime, expected, actual
        ))


if __name__ == "__main__":
    main()
