"use client";

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { resolveApiBaseUrl } from "@/config/api";

const ROLE_OPTIONS = ["ADMIN", "TECHNICIAN", "CUSTOMER"];

interface AdminUser {
  id: string;
  email: string;
  name?: string | null;
  role: string;
  company_name?: string | null;
  contact_name?: string | null;
  contact_phone?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

type SortField = keyof Pick<
  AdminUser,
  "id" | "email" | "name" | "role" | "company_name" | "contact_name" | "contact_phone" | "created_at" | "updated_at"
>;
type SortDirection = "asc" | "desc";

interface FormState {
  name: string;
  email: string;
  role: string;
  password: string;
  companyName: string;
  contactName: string;
  companyPhone: string;
}

const EMPTY_FORM: FormState = {
  name: "",
  email: "",
  role: "TECHNICIAN",
  password: "",
  companyName: "",
  contactName: "",
  companyPhone: "",
};

const BASE_COLUMNS: { label: string; field?: SortField }[] = [
  { label: "ID", field: "id" },
  { label: "Email", field: "email" },
  { label: "Name", field: "name" },
  { label: "Role", field: "role" },
];

const CUSTOMER_EXTRA_COLUMNS: { label: string; field?: SortField }[] = [
  { label: "Company", field: "company_name" },
  { label: "Company Phone", field: "contact_phone" },
  { label: "Primary Contact", field: "contact_name" },
];

const TRAILING_COLUMNS: { label: string; field?: SortField }[] = [
  { label: "Created At", field: "created_at" },
  { label: "Updated At", field: "updated_at" },
  { label: "Actions" },
];

const getTableColumns = (includeCustomerFields: boolean) =>
  [...BASE_COLUMNS, ...(includeCustomerFields ? CUSTOMER_EXTRA_COLUMNS : []), ...TRAILING_COLUMNS];

const formatDate = (value?: string | null) => {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString();
};

const formatText = (value?: string | null) => {
  if (!value) return "—";
  const normalized = value.trim();
  return normalized.length > 0 ? normalized : "—";
};

const getComparableValue = (user: AdminUser, field: SortField) => {
  const value = user[field];
  if (value == null) return "";
  if (field === "created_at" || field === "updated_at") {
    return new Date(value).getTime() || 0;
  }
  return String(value).toLowerCase();
};

export default function AdminUsersPage() {
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loadingTable, setLoadingTable] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [sortField, setSortField] = useState<SortField>("created_at");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedUser, setSelectedUser] = useState<AdminUser | null>(null);

  const [formState, setFormState] = useState<FormState>(EMPTY_FORM);
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [actionSubmitting, setActionSubmitting] = useState(false);

  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [toastType, setToastType] = useState<"success" | "error">("success");

  const apiBaseUrl = useMemo(() => resolveApiBaseUrl(), []);

  const showToast = useCallback((message: string, type: "success" | "error" = "success") => {
    setToastType(type);
    setToastMessage(message);
    window.setTimeout(() => setToastMessage(null), 3000);
  }, []);

  const extractApiError = (detail: unknown): string | null => {
    if (!detail) return null;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((entry) => {
          if (typeof entry === "string") return entry;
          if (entry && typeof entry === "object") {
            const msg = (entry as Record<string, unknown>).msg;
            const loc = (entry as Record<string, unknown>).loc;
            if (msg && typeof msg === "string") {
              if (Array.isArray(loc)) {
                return `${msg} (${loc.join(" → ")})`;
              }
              return msg;
            }
          }
          try {
            return JSON.stringify(entry);
          } catch (error) {
            return String(entry);
          }
        })
        .join("; ");
    }
    if (typeof detail === "object") {
      const nested = (detail as Record<string, unknown>).detail;
      if (nested && nested !== detail) {
        return extractApiError(nested);
      }
      try {
        return JSON.stringify(detail);
      } catch (error) {
        return String(detail);
      }
    }
    return String(detail);
  };

  const fetchUsers = useCallback(
    async (token: string) => {
      setLoadingTable(true);
      setError(null);
      try {
        const response = await fetch(`${apiBaseUrl}/admin/users`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        if (!response.ok) {
          throw new Error(`Failed to load users (${response.status})`);
        }
        const data = await response.json();
        setUsers(Array.isArray(data) ? data : []);
      } catch (err) {
        console.error("Failed to fetch users:", err);
        setError(err instanceof Error ? err.message : "Unable to load users.");
      } finally {
        setLoadingTable(false);
      }
    },
    [apiBaseUrl]
  );

  useEffect(() => {
    try {
      const token = localStorage.getItem("auth_token");
      if (token) {
        setAuthToken(token);
        fetchUsers(token);
      }
    } catch (error) {
      console.warn("Failed to retrieve auth token:", error);
      setError("Unable to access authentication token.");
    }
  }, [fetchUsers]);

  const filteredUsers = useMemo(() => {
    const term = searchTerm.trim().toLowerCase();
    if (!term) return users;
    return users.filter((user) => {
      return (
        user.email.toLowerCase().includes(term) ||
        (user.name ?? "").toLowerCase().includes(term) ||
        user.role.toLowerCase().includes(term) ||
        (user.company_name ?? "").toLowerCase().includes(term) ||
        (user.contact_name ?? "").toLowerCase().includes(term) ||
        (user.contact_phone ?? "").toLowerCase().includes(term)
      );
    });
  }, [users, searchTerm]);

  const sortedUsers = useMemo(() => {
    const sorted = [...filteredUsers];
    sorted.sort((a, b) => {
      const direction = sortDirection === "asc" ? 1 : -1;
      const aValue = getComparableValue(a, sortField);
      const bValue = getComparableValue(b, sortField);
      if (typeof aValue === "number" && typeof bValue === "number") {
        return (aValue - bValue) * direction;
      }
      return String(aValue).localeCompare(String(bValue)) * direction;
    });
    return sorted;
  }, [filteredUsers, sortField, sortDirection]);

  const customerUsers = useMemo(() => sortedUsers.filter((user) => user.role.toUpperCase() === "CUSTOMER"), [sortedUsers]);
  const internalUsers = useMemo(() => sortedUsers.filter((user) => user.role.toUpperCase() !== "CUSTOMER"), [sortedUsers]);

  const renderUserTable = (tableUsers: AdminUser[], emptyMessage: string, includeCustomerFields: boolean) => {
    const columns = getTableColumns(includeCustomerFields);

    if (loadingTable) {
      return (
        <tr>
          <td colSpan={columns.length} className="px-4 py-6 text-center text-muted-foreground">
            Loading users...
          </td>
        </tr>
      );
    }

    if (error) {
      return (
        <tr>
          <td colSpan={columns.length} className="px-4 py-6 text-center text-destructive">
            {error}
          </td>
        </tr>
      );
    }

    if (tableUsers.length === 0) {
      return (
        <tr>
          <td colSpan={columns.length} className="px-4 py-6 text-center text-muted-foreground">
            {emptyMessage}
          </td>
        </tr>
      );
    }

    return tableUsers.map((user) => (
      <tr key={user.id} className="group transition-colors hover:bg-muted/40">
        <td className="whitespace-nowrap px-4 py-3 text-sm text-muted-foreground">{user.id}</td>
        <td className="whitespace-nowrap px-4 py-3 text-sm font-medium">{user.email}</td>
        <td className="whitespace-nowrap px-4 py-3 text-sm">{user.name ?? "—"}</td>
        <td className="whitespace-nowrap px-4 py-3">
          <span className="inline-flex rounded-full border border-border bg-muted/60 px-2 py-0.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {user.role}
          </span>
        </td>
        {includeCustomerFields && (
          <>
            <td className="whitespace-nowrap px-4 py-3 text-sm">{formatText(user.company_name)}</td>
            <td className="whitespace-nowrap px-4 py-3 text-sm">{formatText(user.contact_phone)}</td>
            <td className="whitespace-nowrap px-4 py-3 text-sm">{formatText(user.contact_name)}</td>
          </>
        )}
        <td className="whitespace-nowrap px-4 py-3 text-sm text-muted-foreground">
          {formatDate(user.created_at)}
        </td>
        <td className="whitespace-nowrap px-4 py-3 text-sm text-muted-foreground">
          {formatDate(user.updated_at)}
        </td>
        <td className="whitespace-nowrap px-4 py-3 text-sm">
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleEditUser(user)}
              className="border-border text-xs"
            >
              Edit
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => handleDeleteUser(user)}
              className="text-xs"
            >
              Delete
            </Button>
          </div>
        </td>
      </tr>
    ));
  };

  const handleSort = (field: SortField) => {
    setSortField(field);
    setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
  };

  const resetFormState = () => {
    setFormState(EMPTY_FORM);
    setSelectedUser(null);
    setDeleteConfirmation("");
  };

  const closeAllModals = () => {
    setIsAddModalOpen(false);
    setIsEditModalOpen(false);
    setIsDeleteModalOpen(false);
    resetFormState();
  };

  const handleAddUser = () => {
    resetFormState();
    setIsAddModalOpen(true);
  };

  const handleEditUser = (user: AdminUser) => {
    setSelectedUser(user);
    setFormState({
      name: user.name ?? "",
      email: user.email,
      role: user.role,
      password: "",
      companyName: user.company_name ?? "",
      contactName: user.contact_name ?? user.name ?? "",
      companyPhone: user.contact_phone ?? "",
    });
    setIsEditModalOpen(true);
  };

  const handleDeleteUser = (user: AdminUser) => {
    setSelectedUser(user);
    setDeleteConfirmation("");
    setIsDeleteModalOpen(true);
  };

  const submitAddUser = async () => {
    if (!authToken) return;
    setActionSubmitting(true);
    try {
      const payload: Record<string, unknown> = {
        email: formState.email,
        role: formState.role,
        password: formState.password,
      };
      if (isCustomerRole) {
        payload.name = formState.contactName || undefined;
        payload.company_name = formState.companyName;
        payload.contact_name = formState.contactName;
        payload.contact_phone = formState.companyPhone;
      } else {
        payload.name = formState.name || undefined;
        payload.company_name = null;
        payload.contact_name = null;
        payload.contact_phone = null;
      }
      const response = await fetch(`${apiBaseUrl}/admin/create_user`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${authToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(extractApiError(detail) || "Failed to create user");
      }
      showToast("✅ User created");
      await fetchUsers(authToken);
      closeAllModals();
    } catch (err) {
      console.error("Create user failed:", err);
      showToast(err instanceof Error ? err.message : "Failed to create user", "error");
    } finally {
      setActionSubmitting(false);
    }
  };

  const submitEditUser = async () => {
    if (!authToken || !selectedUser) return;
    setActionSubmitting(true);
    try {
      const payload: Record<string, unknown> = {
        email: formState.email,
        role: formState.role,
      };
      if (formState.password) {
        payload.password = formState.password;
      }
      if (isCustomerRole) {
        payload.name = formState.contactName || null;
        payload.company_name = formState.companyName;
        payload.contact_name = formState.contactName;
        payload.contact_phone = formState.companyPhone;
      } else {
        payload.name = formState.name || null;
        payload.company_name = null;
        payload.contact_name = null;
        payload.contact_phone = null;
      }
      const response = await fetch(`${apiBaseUrl}/admin/edit_user/${selectedUser.id}`, {
        method: "PUT",
        headers: {
          Authorization: `Bearer ${authToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(extractApiError(detail) || "Failed to update user");
      }
      showToast("✅ User updated");
      await fetchUsers(authToken);
      closeAllModals();
    } catch (err) {
      console.error("Update user failed:", err);
      showToast(err instanceof Error ? err.message : "Failed to update user", "error");
    } finally {
      setActionSubmitting(false);
    }
  };

  const submitDeleteUser = async () => {
    if (!authToken || !selectedUser) return;
    if (deleteConfirmation !== "DELETE") {
      showToast("Please type DELETE to confirm", "error");
      return;
    }
    setActionSubmitting(true);
    try {
      const response = await fetch(`${apiBaseUrl}/admin/delete_user/${selectedUser.id}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(extractApiError(detail) || "Failed to delete user");
      }
      showToast("✅ User deleted");
      await fetchUsers(authToken);
      closeAllModals();
    } catch (err) {
      console.error("Delete user failed:", err);
      showToast(err instanceof Error ? err.message : "Failed to delete user", "error");
    } finally {
      setActionSubmitting(false);
    }
  };

  const isCustomerRole = formState.role.toUpperCase() === "CUSTOMER";

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 md:mx-0 md:px-6 xl:mx-auto">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">User Management</h1>
          <p className="text-sm text-muted-foreground">Manage administrator and technician accounts.</p>
        </div>
        <Button className="bg-primary text-primary-foreground" onClick={handleAddUser}>
          + Add User
        </Button>
      </div>

      <section className="rounded-xl border bg-background shadow-sm">
        <div className="flex flex-col gap-4 border-b border-border p-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-lg font-semibold">Customer Users</h2>
            <p className="text-xs text-muted-foreground">Manage customer-facing accounts and permissions.</p>
          </div>
          <div className="flex w-full flex-col gap-2 md:w-auto">
            <span className="text-sm font-medium text-muted-foreground">Search</span>
            <input
              type="text"
              placeholder="Search by name, email, or role..."
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              className="w-full rounded-md border border-border bg-muted/70 px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary md:w-72"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-border">
            <thead className="bg-muted/30">
              <tr>
                {getTableColumns(true).map((column) => (
                  <th
                    key={column.label}
                    scope="col"
                    className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground"
                  >
                    {column.field ? (
                      <button
                        type="button"
                        onClick={() => handleSort(column.field!)}
                        className="flex items-center gap-1 hover:text-foreground"
                      >
                        <span>{column.label}</span>
                        {sortField === column.field && (
                          <span className="text-xs">{sortDirection === "asc" ? "▲" : "▼"}</span>
                        )}
                      </button>
                    ) : (
                      column.label
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border bg-background">
              {renderUserTable(customerUsers, "No customer users found.", true)}
            </tbody>
          </table>
        </div>
        <div className="border-t border-border bg-muted/5 px-4 py-3 text-xs text-muted-foreground">
          Customer users typically represent end-user accounts. Manage customer access and permissions here.
        </div>
      </section>

      <section className="rounded-xl border bg-background shadow-sm">
        <div className="flex flex-col gap-4 border-b border-border p-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-lg font-semibold">Internal Users</h2>
            <p className="text-xs text-muted-foreground">Manage administrator and technician access.</p>
          </div>
          <div className="flex w-full flex-col gap-2 md:w-auto">
            <span className="text-sm font-medium text-muted-foreground">Search</span>
            <input
              type="text"
              placeholder="Search by name, email, or role..."
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              className="w-full rounded-md border border-border bg-muted/70 px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary md:w-72"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-border">
            <thead className="bg-muted/30">
              <tr>
                {getTableColumns(false).map((column) => (
                  <th
                    key={column.label}
                    scope="col"
                    className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground"
                  >
                    {column.field ? (
                      <button
                        type="button"
                        onClick={() => handleSort(column.field!)}
                        className="flex items-center gap-1 hover:text-foreground"
                      >
                        <span>{column.label}</span>
                        {sortField === column.field && (
                          <span className="text-xs">{sortDirection === "asc" ? "▲" : "▼"}</span>
                        )}
                      </button>
                    ) : (
                      column.label
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border bg-background">
              {renderUserTable(internalUsers, "No internal users found.", false)}
            </tbody>
          </table>
        </div>
        <div className="border-t border-border bg-muted/5 px-4 py-3 text-xs text-muted-foreground">
          Internal users include administrators and technicians. Manage system-level access here.
        </div>
      </section>

      {/* Toast */}
      {toastMessage && (
        <div className="fixed right-6 top-6 z-50">
          <div
            className={`rounded-md px-4 py-3 shadow-lg ${
              toastType === "success" ? "bg-emerald-500 text-white" : "bg-red-500 text-white"
            }`}
          >
            {toastMessage}
          </div>
        </div>
      )}

      {/* Modals */}
      {(isAddModalOpen || isEditModalOpen) && (
        <Modal title={isAddModalOpen ? "Add User" : "Edit User"} onClose={closeAllModals}>
          <div className="space-y-4">
            <div className="grid gap-2">
              <label className="text-sm font-medium text-muted-foreground">Account Role</label>
              <select
                value={formState.role}
                onChange={(event) => setFormState((prev) => ({ ...prev, role: event.target.value }))}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
              >
                {ROLE_OPTIONS.map((role) => (
                  <option key={role} value={role}>
                    {role}
                  </option>
                ))}
              </select>
            </div>

            {isCustomerRole ? (
              <div className="space-y-3">
                <div className="grid gap-2">
                  <label className="text-sm font-medium text-muted-foreground">Company Name</label>
                  <input
                    type="text"
                    value={formState.companyName}
                    onChange={(event) => setFormState((prev) => ({ ...prev, companyName: event.target.value }))}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus-border-primary focus:ring-1 focus:ring-primary"
                    placeholder="Enter company name"
                    required
                  />
                </div>
                <div className="grid gap-2">
                  <label className="text-sm font-medium text-muted-foreground">Email</label>
                  <input
                    type="email"
                    value={formState.email}
                    onChange={(event) => setFormState((prev) => ({ ...prev, email: event.target.value }))}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus-border-primary focus:ring-1 focus:ring-primary"
                    placeholder="user@example.com"
                    required
                  />
                </div>
                <div className="grid gap-2">
                  <label className="text-sm font-medium text-muted-foreground">Company Phone</label>
                  <input
                    type="tel"
                    value={formState.companyPhone}
                    onChange={(event) => setFormState((prev) => ({ ...prev, companyPhone: event.target.value }))}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus-border-primary focus:ring-1 focus:ring-primary"
                    placeholder="Enter company phone"
                    required
                  />
                </div>
                <div className="grid gap-2">
                  <label className="text-sm font-medium text-muted-foreground">Primary Contact Name</label>
                  <input
                    type="text"
                    value={formState.contactName}
                    onChange={(event) => setFormState((prev) => ({ ...prev, contactName: event.target.value }))}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus-border-primary focus:ring-1 focus:ring-primary"
                    placeholder="Enter contact name"
                    required
                  />
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="grid gap-2">
                  <label className="text-sm font-medium text-muted-foreground">Name</label>
                  <input
                    type="text"
                    value={formState.name}
                    onChange={(event) => setFormState((prev) => ({ ...prev, name: event.target.value }))}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus-border-primary focus:ring-1 focus:ring-primary"
                    placeholder="Optional name"
                  />
                </div>
                <div className="grid gap-2">
                  <label className="text-sm font-medium text-muted-foreground">Email</label>
                  <input
                    type="email"
                    value={formState.email}
                    onChange={(event) => setFormState((prev) => ({ ...prev, email: event.target.value }))}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus-border-primary focus:ring-1 focus:ring-primary"
                    placeholder="user@example.com"
                    required
                  />
                </div>
              </div>
            )}

            <div className="grid gap-2">
              <label className="text-sm font-medium text-muted-foreground">
                Password {isEditModalOpen && <span className="text-xs text-muted-foreground">(leave blank to keep existing)</span>}
              </label>
              <input
                type="password"
                value={formState.password}
                onChange={(event) => setFormState((prev) => ({ ...prev, password: event.target.value }))}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus-border-primary focus:ring-1 focus:ring-primary"
                placeholder="••••••••"
                required={isAddModalOpen}
              />
            </div>
          </div>

          <div className="mt-6 flex justify-end gap-3">
            <Button variant="outline" onClick={closeAllModals} disabled={actionSubmitting}>
              Cancel
            </Button>
            <Button
              onClick={isAddModalOpen ? submitAddUser : submitEditUser}
              disabled={actionSubmitting}
              className="bg-primary text-primary-foreground"
            >
              {actionSubmitting ? "Saving..." : "Save"}
            </Button>
          </div>
        </Modal>
      )}

      {isDeleteModalOpen && selectedUser && (
        <Modal title="Delete User" onClose={closeAllModals}>
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              This action cannot be undone. Type <span className="font-semibold text-destructive">DELETE</span>{" "}
              to permanently remove <span className="font-medium">{selectedUser.email}</span>.
            </p>
            <input
              type="text"
              value={deleteConfirmation}
              onChange={(event) => setDeleteConfirmation(event.target.value)}
              placeholder="Type DELETE to confirm"
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-destructive focus:ring-1 focus:ring-destructive"
            />
          </div>
          <div className="mt-6 flex justify-end gap-3">
            <Button variant="outline" onClick={closeAllModals} disabled={actionSubmitting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={submitDeleteUser} disabled={actionSubmitting}>
              {actionSubmitting ? "Deleting..." : "Delete"}
            </Button>
          </div>
        </Modal>
      )}
    </div>
  );
}

interface ModalProps {
  title: string;
  children: ReactNode;
  onClose: () => void;
}

function Modal({ title, children, onClose }: ModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-xl border border-border bg-background shadow-xl">
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <h2 className="text-lg font-semibold">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            ✕
          </button>
        </div>
        <div className="px-4 py-5">{children}</div>
      </div>
    </div>
  );
}
