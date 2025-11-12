 "use client";
 
 import { useEffect, useState } from "react";
 import { useRouter } from "next/navigation";
 import { AdminPanel } from "@/components/admin/admin-panel";
 import { Button } from "@/components/ui/button";
 
 export default function AdminPage() {
   const router = useRouter();
   const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
 
   useEffect(() => {
     try {
       const token = localStorage.getItem("auth_token");
       if (!token) {
         setIsAdmin(false);
         router.replace("/");
         return;
       }
 
       const payloadBase64 = token.split(".")[1];
       if (!payloadBase64) {
         setIsAdmin(false);
         router.replace("/");
         return;
       }
 
       const payloadJson = atob(payloadBase64.replace(/-/g, "+").replace(/_/g, "/"));
       const payload = JSON.parse(payloadJson);
 
       if (payload?.role !== "ADMIN") {
         setIsAdmin(false);
         router.replace("/");
         return;
       }
 
       setIsAdmin(true);
     } catch (error) {
       console.warn("Failed to validate token:", error);
       setIsAdmin(false);
       router.replace("/");
     }
   }, [router]);
 
   if (isAdmin === null) {
     return (
       <div className="flex min-h-screen items-center justify-center bg-background">
         <p className="text-muted-foreground">Validating access...</p>
       </div>
     );
   }
 
   if (!isAdmin) {
     return null;
   }
 
   return (
     <div className="min-h-screen bg-background">
       <div className="mx-auto max-w-6xl px-4 py-6">
         <Button variant="ghost" className="mb-4 flex items-center gap-2" onClick={() => router.push("/")}>
           <span className="text-lg">←</span>
           <span>Back to Home</span>
         </Button>
         <AdminPanel />
       </div>
     </div>
   );
 }

