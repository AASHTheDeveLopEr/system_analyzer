#!/usr/bin/env python3
"""The menu of System Analyzer.

Run it:  python menu.py
"""

import os
import platform


def show_menu():
    print()
    print("=" * 38)
    print("           SYSTEM ANALYZER")
    print("=" * 38)
    print("  1. Show my PC specs")
    print("  2. Rate my PC out of 100")
    print("  3. Ask 'can I run X?'")
    print("  4. History of my questions")
    print("  0. Exit")
    print("-" * 38)


def show_specs():
    """Real detection moves to system_one.py later; this is just a first step."""
    print("\n--- detected ---")
    print(f"  OS     : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  CPU    : {platform.processor() or platform.machine()}")
    print(f"  Cores  : {os.cpu_count()} logical")
    print(f"  Python : {platform.python_version()}")
    print("  RAM    : not implemented yet")
    print("  GPU    : not implemented yet")


def rate_pc():
    print("\nRating is not built yet - system_one.py will do that.")


def ask_question():
    question = input("\nWhat do you want to run? > ").strip()
    if not question:
        print("You did not type anything.")
        return None
    print(f"\nYou asked: {question}")
    print("No verdict yet - system_one.py will answer with a confidence percent.")
    return question


def main():
    history = []

    while True:
        show_menu()
        try:
            choice = input("Choose > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if choice == "1":
            show_specs()
        elif choice == "2":
            rate_pc()
        elif choice == "3":
            question = ask_question()
            if question:
                history.append(question)
        elif choice == "4":
            if not history:
                print("\nNo questions yet.")
            else:
                print()
                for i, q in enumerate(history, 1):
                    print(f"  {i}. {q}")
        elif choice in ("0", "q", "exit"):
            print("Bye.")
            break
        else:
            print("That is not on the menu - pick 1, 2, 3, 4 or 0.")


if __name__ == "__main__":
    main()
