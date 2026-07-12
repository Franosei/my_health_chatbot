import { beforeEach, describe, expect, it } from "vitest";
import { getStoredToken, setStoredToken } from "./api";

describe("token storage", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("returns an empty string when no token is stored", () => {
    expect(getStoredToken()).toBe("");
  });

  it("stores and retrieves a token", () => {
    setStoredToken("abc123");
    expect(getStoredToken()).toBe("abc123");
  });

  it("clears the stored token when set with an empty string", () => {
    setStoredToken("abc123");
    setStoredToken("");
    expect(getStoredToken()).toBe("");
  });
});
